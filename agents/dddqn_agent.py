"""
Dueling Double DQN (D3QN / DDDQN) Agent

结合了两种改进:
1. Double DQN: 使用在线网络选择动作，目标网络评估Q值，减少过估计
2. Dueling DQN: 将Q值分解为状态价值V(s)和优势函数A(s,a)
   Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

优势:
- 更好的价值估计：即使在某些状态下动作选择不重要，也能学习状态价值
- 减少Q值过估计
- 更稳定的训练过程
"""

import torch.nn as nn
import torch
import gymnasium as gym
from pvz import config
from copy import deepcopy
import numpy as np
from collections import namedtuple, deque
from .threshold import Threshold

HP_NORM = 1
SUN_NORM = 200


def sum_onehot(grid):
    return torch.cat([(grid == (i + 1)).sum(dim=-1, keepdim=True).float() for i in range(4)], dim=-1)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN 网络架构
    
    将Q值分解为:
    - V(s): 状态价值函数 - 评估处于某个状态有多好
    - A(s,a): 优势函数 - 评估某个动作相比平均动作有多好
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """

    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu', 
                 use_zombienet=True, use_gridnet=True, hidden_size=64):
        super(DuelingQNetwork, self).__init__()
        self.device = device

        # Unwrap environment to access plant_deck
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env

        self.n_inputs = config.N_LANES * config.LANE_LENGTH + config.N_LANES + len(unwrapped_env.plant_deck) + 1
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.hidden_size = hidden_size

        # Optional: ZombieNet for processing zombie grid
        self.use_zombienet = use_zombienet
        if use_zombienet:
            self.zombienet_output_size = 1
            self.zombienet = ZombieNet(output_size=self.zombienet_output_size)
            self.n_inputs += (self.zombienet_output_size - 1) * config.N_LANES

        # Optional: GridNet for processing plant grid
        self.use_gridnet = use_gridnet
        if use_gridnet:
            self.gridnet_output_size = 4
            self.gridnet = nn.Linear(self._grid_size, self.gridnet_output_size)
            self.n_inputs += self.gridnet_output_size - self._grid_size

        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(self.n_inputs, hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LeakyReLU()
        )

        # 状态价值流 V(s) - 输出单一标量
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, 1, bias=True)
        )

        # 优势函数流 A(s,a) - 输出每个动作的优势值
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, self.n_outputs, bias=True)
        )

        # Set to GPU if cuda is specified
        if self.device == 'cuda':
            self.feature_layer.cuda()
            self.value_stream.cuda()
            self.advantage_stream.cuda()
            if use_zombienet:
                self.zombienet.cuda()
            if use_gridnet:
                self.gridnet.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )

    def forward(self, state_t):
        """前向传播，返回Q值"""
        features = self.feature_layer(state_t)
        
        # 计算V(s)和A(s,a)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        # 减去均值使得优势函数可辨识（identifiable）
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values

    def decide_action(self, state, mask, epsilon):
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

    def get_qvals(self, state):
        """获取Q值，支持单个状态和批量状态"""
        if type(state) is tuple:
            # 批量处理
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[:, self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:, :self._grid_size]
            
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1, self.zombienet_output_size * config.N_LANES)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1).view(-1, config.N_LANES)
            
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            
            state_t = torch.cat([plant_grid, zombie_grid, state_t[:, 2 * self._grid_size:]], axis=1)
        else:
            # 单个状态
            state_t = torch.FloatTensor(state).to(device=self.device)
            zombie_grid = state_t[self._grid_size:(2 * self._grid_size)].reshape(-1, config.LANE_LENGTH)
            plant_grid = state_t[:self._grid_size]
            
            if self.use_zombienet:
                zombie_grid = self.zombienet(zombie_grid).view(-1)
            else:
                zombie_grid = torch.sum(zombie_grid, axis=1)
            
            if self.use_gridnet:
                plant_grid = self.gridnet(plant_grid)
            
            state_t = torch.cat([plant_grid, zombie_grid, state_t[2 * self._grid_size:]])
        
        return self.forward(state_t)


class ZombieNet(nn.Module):
    """处理僵尸网格的小网络"""
    def __init__(self, output_size=1, hidden_size=5):
        super(ZombieNet, self).__init__()
        self.fc1 = nn.Linear(config.LANE_LENGTH, output_size)

    def forward(self, x):
        return self.fc1(x)


class D3QNAgent:
    """
    Dueling Double DQN Agent (D3QN)
    
    结合:
    - Double DQN: 在线网络选动作，目标网络评估
    - Dueling DQN: V(s) + A(s,a) 架构
    """

    def __init__(self, env, network, buffer, n_iter=100000, batch_size=32):
        self._grid_size = config.N_LANES * config.LANE_LENGTH
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        
        # Epsilon衰减策略
        self.threshold = Threshold(
            seq_length=n_iter, 
            start_epsilon=1.0, 
            interpolation="exponential",
            end_epsilon=0.1
        )
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        self.player = PlayerQ(env=env, render=False)

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
                action = 0  # Do nothing
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
              network_sync_frequency=2000,
              evaluate_frequency=5000,
              evaluate_n_iter=1000):
        """训练D3QN agent"""
        
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
                
                # 同步目标网络
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
                    
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                        ep, mean_rewards, mean_iteration), end="")

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
                        avg_score, avg_iter = _evaluate(self.player, self.network, n_iter=evaluate_n_iter, verbose=False)
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)
                        
                        if self.network.device == 'cuda':
                            torch.cuda.empty_cache()

    def calculate_loss(self, batch):
        """
        计算Dueling Double DQN损失
        
        Double DQN: 用在线网络选择动作，用目标网络评估Q值
        """
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).to(device=self.network.device).reshape(-1, 1)
        dones_t = torch.BoolTensor(dones).to(device=self.network.device)

        # 当前状态的Q值
        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)

        # Double DQN: 在线网络选动作，目标网络评估
        with torch.no_grad():
            next_masks = np.array([self._get_mask(s) for s in next_states])
            
            # 用在线网络选择最佳动作
            qvals_next_pred = self.network.get_qvals(next_states)
            qvals_next_pred[np.logical_not(next_masks)] = qvals_next_pred.min()
            next_actions = torch.max(qvals_next_pred, dim=-1)[1]
            next_actions_t = next_actions.reshape(-1, 1)
            
            # 用目标网络评估选中动作的Q值
            target_qvals = self.target_network.get_qvals(next_states)
            qvals_next = torch.gather(target_qvals, 1, next_actions_t)

        qvals_next[dones_t] = 0  # 终止状态Q值为0
        expected_qvals = self.gamma * qvals_next + rewards_t
        
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        """执行一次网络更新"""
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())
        
        del batch, loss

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

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1) / HP_NORM

    def _save_training_data(self, nn_name):
        """保存训练数据"""
        np.save(nn_name + "_rewards", self.training_rewards)
        np.save(nn_name + "_iterations", self.training_iterations)
        np.save(nn_name + "_real_rewards", self.real_rewards)
        np.save(nn_name + "_real_iterations", self.real_iterations)
        torch.save(self.training_loss, nn_name + "_loss")

    def initialize(self):
        """初始化训练状态"""
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        obs_raw, _info = self.env.reset()
        self.s_0 = self._transform_observation(obs_raw)


class experienceReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                  field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class PlayerQ:
    """用于评估的玩家类"""
    
    def __init__(self, env=None, render=True):
        if env is None:
            self.env = gym.make('gym_pvz:pvz-env-v2')
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
        observation = observation.astype(np.float64)
        observation = np.concatenate([
            observation[:self._grid_size],
            observation[self._grid_size:(2 * self._grid_size)] / HP_NORM,
            [observation[2 * self._grid_size] / SUN_NORM],
            observation[2 * self._grid_size + 1:]
        ])
        return observation

    def _grid_to_lane(self, grid):
        grid = np.reshape(grid, (config.N_LANES, config.LANE_LENGTH))
        return np.sum(grid, axis=1) / HP_NORM

    def play(self, agent, epsilon=0):
        """执行一局游戏并收集数据"""
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

    def get_render_info(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env._scene._render_info
