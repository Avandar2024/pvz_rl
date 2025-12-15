import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from pvz import config

HP_NORM = 100
SUN_NORM = 200
# ------------------------------------------

# 1. 核心网络模型 (Actor-Critic Network)
class ActorCriticNetwork(nn.Module):
    """
    一个具有共享主干、分离的 Actor (策略) 和 Critic (价值) 头部的网络。
    """
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享主干: 3x512
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Actor Head (策略): 2x256 -> output_size (动作概率)
        self.actor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size) # Logits, Softmax applied in forward/Categorical
        )
        
        # Critic Head (价值): 3x256 -> 1 (状态价值 V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.shared_trunk(x)
        # Actor 输出 logits，便于数值稳定和 Categorical 采样
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value

    def get_value(self, x):
        features = self.shared_trunk(x)
        return self.critic_head(features)

# 2. PPO Agent (策略梯度算法)
class PPOAgent():
    """
    Proximal Policy Optimization (PPO) Agent 的实现。
    支持单环境或向量化环境的数据处理。
    """
    def __init__(self, input_size, possible_actions, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, gae_lambda=0.95, mini_batch_size=256):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(input_size, len(possible_actions)).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        
        self.reset_storage()

    def reset_storage(self):
        """清除经验缓冲区"""
        self.saved_states = []
        self.saved_actions = []
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_dones = []
        self.saved_values = []

    @torch.no_grad()
    def decide_action(self, state):
        """
        根据当前状态决定动作。
        state 可以是单个 np.array/torch.Tensor 或一批。
        """
        # 将输入状态转换为 Tensor，并移动到设备上
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        elif hasattr(state, 'get'): # 处理 CuPy 数组
            state_tensor = torch.as_tensor(state.get(), device=self.device, dtype=torch.float32)
        else: # 假设是 Tensor
            state_tensor = state.to(self.device).float()
            
        # 确保输入是批处理格式 (B, D)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        action_logits, value = self.network(state_tensor)
        # Clone value to avoid CUDA Graph memory overwrite issues
        value = value.clone().squeeze(-1)
        
        m = Categorical(logits=action_logits)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        # 存储经验。如果是单环境，需要解包
        is_single_env = state_tensor.size(0) == 1
        
        self.saved_states.append(state_tensor if not is_single_env else state_tensor[0])
        self.saved_actions.append(action if not is_single_env else action[0])
        self.saved_log_probs.append(log_prob if not is_single_env else log_prob[0])
        self.saved_values.append(value if not is_single_env else value[0])
        
        # 返回 numpy 格式的动作 (兼容 gym.step)
        return action.cpu().numpy()
    
    def store_reward_done(self, reward, done):
        """存储奖励和终止信号 (支持向量化环境)"""
        # 统一转换为 Tensor 并存储
        self.saved_rewards.append(torch.tensor(reward, device=self.device, dtype=torch.float32))
        self.saved_dones.append(torch.tensor(done, device=self.device, dtype=torch.float32))

    def update(self, next_observation):
        """
        PPO 学习步：计算 GAE 和 Returns，执行 K_epochs 次优化。
        
        Args:
            next_observation (np.ndarray/Tensor): 下一个状态或一批下一个状态 (用于计算终值 V(s')).
        """
        # 1. 计算下一个状态的价值 V(s')
        if hasattr(next_observation, 'get'):
             next_observation = next_observation.get()
             
        if isinstance(next_observation, np.ndarray):
            next_obs_tensor = torch.as_tensor(next_observation, device=self.device, dtype=torch.float32)
        else:
            next_obs_tensor = next_observation.to(self.device).float()

        with torch.no_grad():
            next_value = self.network.get_value(next_obs_tensor).squeeze(-1)
        
        # 2. 准备缓冲区数据
        # Stack all collected tensors
        old_states = torch.stack(self.saved_states)
        old_actions = torch.stack(self.saved_actions)
        old_log_probs = torch.stack(self.saved_log_probs)
        old_values = torch.stack(self.saved_values)
        rewards = torch.stack(self.saved_rewards)
        dones = torch.stack(self.saved_dones)
        
        # 3. 计算 GAE (广义优势估计)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0
        num_steps = len(rewards)
        
        # 处理向量化环境的情况：形状为 (Time, NumEnvs, ...)
        for t in reversed(range(num_steps)):
            # V_{t+1}，如果是最后一个时间步，使用 next_value
            V_next = next_value if t == num_steps - 1 else old_values[t+1]
            non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * V_next * non_terminal - old_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae

        returns = advantages + old_values
        
        # 4. 扁平化数据以进行 PPO 优化
        b_states = old_states.view(-1, old_states.shape[-1])
        b_actions = old_actions.view(-1)
        b_log_probs = old_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        
        # 5. 优势归一化
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
        # 6. PPO 梯度更新
        dataset_size = b_states.size(0)
        total_loss = 0
        n_updates = 0
        
        for _ in range(self.K_epochs):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # 提取 Mini-batch
                batch_states = b_states[batch_indices]
                batch_actions = b_actions[batch_indices]
                batch_log_probs = b_log_probs[batch_indices]
                batch_advantages = b_advantages[batch_indices]
                batch_returns = b_returns[batch_indices]
                
                # 前向传播 (新策略)
                action_logits, state_values = self.network(batch_states)
                state_values = state_values.squeeze(-1)
                m = Categorical(logits=action_logits)
                log_probs = m.log_prob(batch_actions)
                entropy = m.entropy()
                
                # PPO Clip Loss
                ratios = torch.exp(log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                # 损失函数: Actor Loss (最小化) + 0.5 * Critic Loss (MSE) - 0.01 * Entropy Loss (最大化)
                # 注: 0.5 * Critic Loss 是一种常见的平衡权重
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(state_values, batch_returns)
                entropy_loss = -0.01 * entropy.mean()
                
                loss = policy_loss + value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # 可以添加梯度裁剪 (可选)
                # nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
                
        self.reset_storage()
        return total_loss / n_updates if n_updates > 0 else 0

    def save(self, nn_name):
        torch.save(self.network.state_dict(), nn_name)

    def load(self, nn_name):
        self.network.load_state_dict(torch.load(nn_name, map_location=self.device))

# 3. 训练器 (Trainer)
class Trainer():
    """环境交互和数据预处理的逻辑封装"""
    def __init__(self, render=False, max_frames=1000):
        # 使用 gym.make() 创建环境
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render
        self._grid_size = config.N_LANES * config.LANE_LENGTH

        # 剥离可能存在的 Wrapper 以获取原始环境属性
        self.env_base = self.env.unwrapped
        
    def compile_agent_network(self, agent):
        """编译 PPO Agent 中的 ActorCriticNetwork"""
        # 
        if hasattr(agent.network, 'shared_trunk') and torch.cuda.is_available():
             print("Compiling ActorCriticNetwork for performance...")
             agent.network = torch.compile(agent.network, fullgraph=True, mode="reduce-overhead")

    def num_observations(self):
        """返回转换后的观察空间维度"""
        # (植物网格) + (僵尸车道总血量) + (阳光值) + (卡牌组信息)
        return self._grid_size + config.N_LANES + len(self.env_base.plant_deck) + 1

    def num_actions(self):
        """返回动作空间维度"""
        return self.env.action_space.n

    def get_render_info(self):
        return self.env_base._scene._render_info

    def _transform_observation(self, observation):
        """
        对原始观察进行归一化和特征提取。
        原始观察假设为 1D NumPy 数组：[Plant_Grid, Zombie_Grid, Sun, Deck...]
        """
        # 将输入观察转换为 NumPy 数组
        if hasattr(observation, 'get'): # 处理 CuPy 数组
            observation = observation.get()
        
        observation = np.asarray(observation, dtype=np.float32)
        
        # 1. 植物网格 ([:_grid_size]) - 保持不变
        plant_grid = observation[:self._grid_size]
        
        # 2. 僵尸网格 (_grid_size: 2*_grid_size) -> 僵尸血量和（按车道）
        zombie_grid = observation[self._grid_size:2*self._grid_size]
        # 重塑为 (N_LANES, LANE_LENGTH)
        zombie_grid = zombie_grid.reshape((config.N_LANES, config.LANE_LENGTH))
        # 按行求和 (每车道的血量总和) 并归一化
        zombie_lane_sum = np.sum(zombie_grid, axis=1) / HP_NORM
        
        # 3. 阳光值 (2*_grid_size) - 归一化
        sun_val = observation[2 * self._grid_size] / SUN_NORM
        
        # 4. 卡牌信息 (2*_grid_size+1:) - 保持不变
        card_info = observation[2 * self._grid_size+1:]
        
        # 拼接成新的观察向量
        new_observation = np.concatenate([
            plant_grid,
            zombie_lane_sum,
            [sun_val],
            card_info
        ]).astype(np.float32)
        
        return new_observation

    def play(self, agent):
        """
        进行一集游戏并与 Agent 交互。
        此方法专为单环境交互设计。
        """
        # Gymnasium reset() 返回 (observation, info)
        observation, _info = self.env.reset()
        observation = self._transform_observation(observation)

        episode_steps = 0
        episode_reward = 0.0

        while self.env_base._scene._chrono < self.max_frames:
            if self.render:
                self.env.render()

            # Agent 决定动作
            action = agent.decide_action(observation)
            
            # 环境交互 (observation, reward, terminated, truncated, info)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            next_observation = self._transform_observation(next_observation)
            
            # 存储奖励和终止状态 (PPOVecAgent 的接口复用)
            # 对于单环境，reward 和 done 必须是标量或形状为 (1,) 的数组
            agent.store_reward_done(np.array([reward]), np.array([done]))

            observation = next_observation
            episode_reward += reward
            episode_steps += 1

            if done:
                break
        
        # PPO 更新需要最后一个状态的价值 V(s')
        # 如果是 terminated 或 truncated，V(s') 应该为 0
        if done:
            next_value= np.zeros(1, dtype=np.float32)
        else:
            # 使用下一个观察值 (即当前循环中最后一个 next_observation) 来估算 V(s')
            # PPOAgent.update 会处理这个计算
            next_value = next_observation 

        # 触发 PPO 更新
        agent.update(next_value)
        
        return episode_steps, episode_reward
