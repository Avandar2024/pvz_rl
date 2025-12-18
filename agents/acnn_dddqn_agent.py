"""
ACNN版本的 Dueling Double DQN (ACNN-D3QN) Agent

在CNN-D3QN基础上添加CBAM注意力机制，让网络学会聚焦于关键区域。

改动点：
1. 使用ACNNFeatureExtractor替代CNNFeatureExtractor
2. 网络结构增加CBAM模块
3. 其他逻辑与CNN-D3QN完全一致
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from copy import deepcopy
from pvz import config

from .threshold import Threshold
from .acnn_networks import ACNNFeatureExtractor

# 归一化常量
HP_NORM = 1000
SUN_NORM = 1000


class ACNNDuelingQNetwork(nn.Module):
    """
    ACNN版本的Dueling Q网络
    
    在CNN基础上添加CBAM注意力机制:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(self, env, epsilon=0.05, learning_rate=1e-4, device='cpu',
                 hidden_channels=32, feature_size=128,
                 attention_reduction=4, attention_kernel=3, use_residual=True):
        super(ACNNDuelingQNetwork, self).__init__()
        self.device = device

        # 获取环境信息
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        
        n_plant_types = len(unwrapped_env.plant_deck)
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        
        # ACNN特征提取器（带CBAM注意力）
        self.feature_extractor = ACNNFeatureExtractor(
            n_plant_types=n_plant_types,
            hidden_channels=hidden_channels,
            output_features=feature_size,
            attention_reduction=attention_reduction,
            attention_kernel=attention_kernel,
            use_residual=use_residual
        )
        
        # Dueling架构: Value流
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_size // 2, 1)
        )
        
        # Dueling架构: Advantage流
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_size // 2, self.n_outputs)
        )
        
        self._initialize_weights()
        self.to(device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.epsilon = epsilon

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播，Dueling架构"""
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def get_qvals(self, state):
        """获取Q值，与原版接口一致"""
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        features = self.feature_extractor(state_t)
        q_values = self.forward(features)
        
        if type(state) is not tuple:
            q_values = q_values.squeeze(0)
        
        return q_values

    def decide_action(self, state, mask, epsilon):
        """选择动作，与原版接口一致"""
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action

    def get_greedy_action(self, state, mask):
        with torch.no_grad():
            qvals = self.get_qvals(state)
            qvals[np.logical_not(mask)] = qvals.min()
            return torch.max(qvals, dim=-1)[1].item()


class ACNN_D3QNAgent:
    """
    ACNN版本的D3QN Agent
    
    训练逻辑、输出格式与CNN-D3QN完全一致，只是网络换成ACNN版本。
    """

    def __init__(self, env, network, buffer, n_iter=100000, batch_size=64,
                 tau=0.001, grad_clip=10.0, end_epsilon=0.15):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        
        # 软更新参数
        self.tau = tau
        self.grad_clip = grad_clip
        
        # Epsilon衰减策略
        self.threshold = Threshold(
            seq_length=int(n_iter * 0.8),
            start_epsilon=1.0, 
            interpolation="exponential",
            end_epsilon=end_epsilon
        )
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        self.player = PlayerQ_ACNN(env=env, render=False)
        
        # 最佳模型追踪
        self.best_eval_score = float('-inf')
        self.best_model_state = None
        self.best_episode = 0
        self.best_wins = 0
        self.best_losses = 0
        self.best_timeouts = 0
        self.best_total_games = 0

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.eval_stats = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        
        obs_raw, _info = self.env.reset()
        self.s_0 = self._transform_observation(obs_raw)

    def take_step(self, mode='train'):
        """执行一步动作"""
        def _inner_env(e):
            env = e
            while hasattr(env, 'env'):
                env = env.env
            return env

        inner = _inner_env(self.env)
        mask = np.array(inner.mask_available_actions())
        
        if mode == 'explore':
            if np.random.random() < 0.5:
                action = 0
            else:
                action = np.random.choice(np.arange(inner.action_space.n)[mask])
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1

        obs_raw, r, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        s_1 = self._transform_observation(obs_raw)

        self.rewards += r
        self.buffer.append(self.s_0, action, r, done, s_1)
        self.s_0 = s_1.copy()
        
        if done:
            if mode != "explore":
                self.training_iterations.append(min(config.MAX_FRAMES, _inner_env(self.env)._scene._chrono))
            obs_raw, _info = self.env.reset()
            self.s_0 = self._transform_observation(obs_raw)
        return done

    def train(self, gamma=0.99, max_episodes=100000,
              network_update_frequency=32,
              network_sync_frequency=10000,
              evaluate_frequency=5000,
              evaluate_n_iter=1000):
        """训练循环"""
        
        self.gamma = gamma
        
        # 预填充经验回放缓冲区
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')

        ep = 0
        training = True

        while training:
            self.rewards = 0
            done = False
            
            while not done:
                self.epsilon = self.threshold.epsilon(ep)
                done = self.take_step(mode='train')
                
                # 更新网络
                if self.step_count % network_update_frequency == 0:
                    self.update()
                
                # 周期性硬同步
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss) if len(self.update_loss) > 0 else 0.0)
                    self.update_loss = []
                    
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    
                    mean_iteration = np.mean(self.training_iterations[-self.window:])
                    self.mean_training_iterations.append(mean_iteration)
                    
                    current_ep_idx = min(ep, self.threshold.seq_length - 1)
                    current_epsilon = self.threshold.epsilon(current_ep_idx)
                    
                    # 输出格式与原版完全一致
                    status = f"Ep {ep:6d} | Mean Reward {mean_rewards:7.2f} | Mean Iter {mean_iteration:6.1f} | ε {current_epsilon:.3f}"
                    print(f"\r{status:80s}", end="")

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(ep))
                        break
                    
                    # 定期评估
                    if (ep % evaluate_frequency) == evaluate_frequency - 1:
                        from .evaluate_agent import evaluate as _evaluate
                        avg_score, avg_iter, win_rate, loss_rate, timeout_rate, wins, losses, timeouts = _evaluate(
                            self.player, self.network, n_iter=evaluate_n_iter, verbose=False)
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)
                        self.eval_stats.append((wins, losses, timeouts, evaluate_n_iter))
                        
                        # 打印评估结果
                        print(f"\n{'='*60}")
                        print(f" Evaluation @ Episode {ep+1}")
                        print(f" Avg Score: {avg_score:.2f} | Avg Frames: {avg_iter:.1f}")
                        print(f" Win: {win_rate:.1f}% ({wins}/{evaluate_n_iter}) | Loss: {loss_rate:.1f}% ({losses}/{evaluate_n_iter}) | Timeout: {timeout_rate:.1f}% ({timeouts}/{evaluate_n_iter})")
                        
                        # 保存最佳模型
                        if avg_score > self.best_eval_score:
                            self.best_eval_score = avg_score
                            self.best_model_state = deepcopy(self.network.state_dict())
                            self.best_episode = ep + 1
                            self.best_wins = wins
                            self.best_losses = losses
                            self.best_timeouts = timeouts
                            self.best_total_games = evaluate_n_iter
                            print(f" ★ NEW BEST MODEL! Score: {avg_score:.2f}")
                        print(f"{'='*60}\n")
                        
                        if self.network.device == 'cuda':
                            torch.cuda.empty_cache()

    def calculate_loss(self, batch):
        """计算Double DQN损失（使用Huber Loss）"""
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).to(device=self.network.device).reshape(-1, 1)
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)

        with torch.no_grad():
            next_masks = np.array([self._get_mask(s) for s in next_states])
            
            qvals_next_pred = self.network.get_qvals(next_states)
            qvals_next_pred[np.logical_not(next_masks)] = qvals_next_pred.min()
            next_actions = torch.max(qvals_next_pred, dim=-1)[1]
            next_actions_t = next_actions.reshape(-1, 1)
            
            target_qvals = self.target_network.get_qvals(next_states)
            qvals_next = torch.gather(target_qvals, 1, next_actions_t)

        qvals_next[dones_t] = 0
        expected_qvals = self.gamma * qvals_next + rewards_t
        
        loss = nn.SmoothL1Loss()(qvals, expected_qvals)
        return loss

    def update(self):
        """执行一次网络更新"""
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        
        self.network.optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target()
        
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        
        del batch, loss
    
    def _soft_update_target(self):
        """软更新目标网络"""
        for target_param, online_param in zip(self.target_network.parameters(), 
                                               self.network.parameters()):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def _transform_observation(self, observation):
        """转换观察值"""
        observation = observation.astype(np.float64)
        observation = np.concatenate([
            observation[:self._grid_size],
            observation[self._grid_size:(2 * self._grid_size)] / HP_NORM,
            [observation[2 * self._grid_size] / SUN_NORM],
            observation[2 * self._grid_size + 1:]
        ])
        return observation

    def _get_mask(self, observation):
        """获取可用动作掩码"""
        empty_cells = np.nonzero((observation[:self._grid_size] == 0).reshape(config.N_LANES, config.LANE_LENGTH))
        env_for_space = self.env
        while hasattr(env_for_space, 'env'):
            env_for_space = env_for_space.env
        
        mask = np.zeros(env_for_space.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * len(env_for_space.plant_deck)

        available_plants = observation[-len(env_for_space.plant_deck):]
        for i in range(len(available_plants)):
            if available_plants[i]:
                idx = empty_cells + i + 1
                mask[idx] = True
        return mask

    def _save_training_data(self, nn_name):
        """保存训练数据"""
        np.save(nn_name + "_rewards", self.training_rewards)
        np.save(nn_name + "_iterations", self.training_iterations)
        np.save(nn_name + "_real_rewards", self.real_rewards)
        np.save(nn_name + "_real_iterations", self.real_iterations)
        np.save(nn_name + "_eval_stats", self.eval_stats)
        torch.save(self.training_loss, nn_name + "_loss")
        
        # 保存最佳模型
        if self.best_model_state is not None:
            best_model_path = nn_name + "_best"
            self.network.load_state_dict(self.best_model_state)
            torch.save(self.network, best_model_path)
            print(f"\n★ Best model saved: {best_model_path}")
            print(f"  From episode {self.best_episode}, Score: {self.best_eval_score:.2f}")
            print(f"  Win: {self.best_wins}/{self.best_total_games}, Loss: {self.best_losses}/{self.best_total_games}")


class PlayerQ_ACNN:
    """ACNN版本的Player，用于评估"""
    
    def __init__(self, env=None, render=False):
        if env is None:
            self.env = gym.make('gym_pvz:pvz-env-v3')
        else:
            self.env = env
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def get_actions(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return list(range(env.action_space.n))

    def num_observations(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(env.plant_deck) + 1

    def num_actions(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env.action_space.n

    def _transform_observation(self, observation):
        """观察转换"""
        observation = observation.astype(np.float64)
        observation = np.concatenate([
            observation[:self._grid_size],
            observation[self._grid_size:(2 * self._grid_size)] / HP_NORM,
            [observation[2 * self._grid_size] / SUN_NORM],
            observation[2 * self._grid_size + 1:]
        ])
        return observation

    def play(self, agent, epsilon=0):
        """玩一局游戏"""
        summary = dict()
        summary['rewards'] = list()
        summary['observations'] = list()
        summary['actions'] = list()
        
        reset_res = self.env.reset()
        if isinstance(reset_res, tuple):
            obs_raw = reset_res[0]
        else:
            obs_raw = reset_res
        observation = self._transform_observation(obs_raw)

        while True:
            if self.render:
                self.env.render()
            
            env_for_mask = self.env
            while hasattr(env_for_mask, 'env'):
                env_for_mask = env_for_mask.env
            try:
                mask = env_for_mask.mask_available_actions()
            except Exception:
                mask = np.full(self.num_actions(), True)
            
            action = agent.decide_action(observation, mask, epsilon)
            summary['observations'].append(observation)
            summary['actions'].append(action)
            
            step_res = self.env.step(action)
            if isinstance(step_res, tuple):
                if len(step_res) == 5:
                    obs_raw, reward, terminated, truncated, info = step_res
                    done = bool(terminated or truncated)
                elif len(step_res) == 4:
                    obs_raw, reward, done, info = step_res
                else:
                    raise RuntimeError(f"Unexpected env.step() return shape: {len(step_res)}")
            else:
                raise RuntimeError("env.step() did not return a tuple")
            
            observation = self._transform_observation(obs_raw)
            summary['rewards'].append(reward)

            if done:
                break

        summary['observations'] = np.vstack(summary['observations'])
        summary['actions'] = np.vstack(summary['actions'])
        summary['rewards'] = np.vstack(summary['rewards'])
        return summary


# 经验回放缓冲区
class experienceReplayBuffer:
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = [[None] for _ in range(memory_size)]
        self.index = 0
        self.full = False

    def append(self, s_0, a, r, d, s_1):
        self.buffer[self.index] = [s_0, a, r, d, s_1]
        self.index = (self.index + 1) % self.memory_size
        if self.index == 0:
            self.full = True

    def sample_batch(self, batch_size=32):
        if self.full:
            indices = np.random.choice(self.memory_size, size=batch_size, replace=False)
        else:
            indices = np.random.choice(self.index, size=batch_size, replace=False)
        
        states = tuple([self.buffer[i][0] for i in indices])
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        dones = [self.buffer[i][3] for i in indices]
        next_states = tuple([self.buffer[i][4] for i in indices])
        
        return states, actions, rewards, dones, next_states

    def burn_in_capacity(self):
        if self.full:
            return 1.0
        return self.index / self.burn_in
