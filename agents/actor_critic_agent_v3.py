import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# 假设 config 模块存在
# from pvz import config
# 临时定义 config 中必要的变量以使代码可运行
class Config:
    N_LANES = 5
    LANE_LENGTH = 9
    plant_deck = [1, 2, 3, 4] # 4种植物
config = Config()

HP_NORM = 100
SUN_NORM = 200
# ------------------------------------------

def get_device():
    """自动选择最佳设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 针对 Apple Silicon 的性能优化
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 1. 核心网络模型 (Actor-Critic Network)
class ActorCriticNetwork(nn.Module):
    """
    一个具有共享主干、分离的 Actor (策略) 和 Critic (价值) 头部的网络。
    """
    def __init__(self, base_dim=3, plant_dim=4, location_dim=45):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享主干: 使用 nn.Sequential 简化
        self.shared_trunk = nn.Sequential(
            # 使用 nn.LazyLinear 允许第一次 forward 自动确定输入维度
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Actor Heads
        self.actor_base = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, base_dim))
        self.actor_plant = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, plant_dim))
        self.actor_location = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, location_dim))
        
        # Critic Head (价值)
        self.critic_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.shared_trunk(x)
        base_logits = self.actor_base(features)
        plant_logits = self.actor_plant(features)
        location_logits = self.actor_location(features)
        state_value = self.critic_head(features)
        return base_logits, plant_logits, location_logits, state_value

    def get_value(self, x):
        features = self.shared_trunk(x)
        return self.critic_head(features)

# 2. PPO Agent (策略梯度算法)
class PPOAgent():
    """Proximal Policy Optimization (PPO) Agent 的实现。"""
    def __init__(self, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, gae_lambda=0.95, mini_batch_size=256):
        
        self.device = get_device()
        print(f"Using device: {self.device}")

        # 固定的输出维度
        self.base_dim = 3
        self.plant_dim = 4
        self.location_dim = config.N_LANES * config.LANE_LENGTH
        self.n_lanes = config.N_LANES
        self.lane_length = config.LANE_LENGTH

        self.network = ActorCriticNetwork(self.base_dim, self.plant_dim, self.location_dim).to(self.device)
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

    def _to_tensor(self, data, dtype=torch.float32, is_action=False):
        """统一的数据转换和设备移动函数，处理 np.ndarray, CuPy, Tensor"""
        if hasattr(data, 'get'): # CuPy
            data = data.get()
        
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, (int, float)):
             tensor = torch.tensor(data, dtype=dtype)
        else: # 假设是 Tensor
            tensor = data
        
        if is_action:
             # 动作应为 long 类型
            return tensor.long().to(self.device)
        return tensor.to(self.device).to(dtype)

    @torch.no_grad()
    def decide_action(self, state):
        """
        根据当前状态决定动作。
        state 可以是单个 np.array/torch.Tensor 或一批。
        """
        state_tensor = self._to_tensor(state)
        
        # 确保输入是批处理格式 (B, D)
        is_single_env = state_tensor.dim() == 1
        if is_single_env:
            state_tensor = state_tensor.unsqueeze(0)
            
        base_logits, plant_logits, loc_logits, value = self.network(state_tensor)
        value = value.squeeze(-1) # (B, 1) -> (B,)
        
        # Sample from each head
        m_base = Categorical(logits=base_logits)
        action_base = m_base.sample()
        log_prob_base = m_base.log_prob(action_base)

        m_plant = Categorical(logits=plant_logits)
        action_plant = m_plant.sample()
        log_prob_plant = m_plant.log_prob(action_plant)

        m_loc = Categorical(logits=loc_logits)
        action_loc = m_loc.sample()
        log_prob_loc = m_loc.log_prob(action_loc)

        # 结合 log_prob (只有 Base==2, 即 Place Plant, 时，Plant/Location 才起作用)
        mask_place = (action_base == 2).float()
        combined_log_prob = log_prob_base + mask_place * (log_prob_plant + log_prob_loc)

        # Stack actions for storage as a tuple (base, plant, loc)
        action_tuple = torch.stack([action_base, action_plant, action_loc], dim=-1)

        # 存储经验。如果是单环境，需要存储 (D,) 的 Tensor
        self.saved_states.append(state_tensor if not is_single_env else state_tensor[0])
        self.saved_actions.append(action_tuple if not is_single_env else action_tuple[0])
        self.saved_log_probs.append(combined_log_prob if not is_single_env else combined_log_prob[0])
        self.saved_values.append(value if not is_single_env else value[0])

        # 映射到环境动作整数
        def map_action(b, p, l):
            if b == 2:  # Place Plant
                loc_index = int(l)
                lane = loc_index % self.n_lanes
                pos = loc_index // self.n_lanes
                # 动作空间映射逻辑: 1 + (位置 * N_PLANETS + 植物类型)
                return 1 + (pos * self.n_lanes + lane) * self.plant_dim + int(p)
            else:
                return 0 # No-Op (包括 No-Op 和 Collect Sun)

        # 转换为 numpy/python scalar 传给环境
        a_base = action_base.cpu().numpy()
        a_plant = action_plant.cpu().numpy()
        a_loc = action_loc.cpu().numpy()

        if is_single_env:
            return map_action(a_base.item(), a_plant.item(), a_loc.item())
        else:
            # 使用列表推导式高效映射向量环境的动作
            return np.array([map_action(b, p, l) for b, p, l in zip(a_base, a_plant, a_loc)])
        
    def store_reward_done(self, reward, done):
        """存储奖励和终止信号 (支持单环境或向量化环境)"""
        # 统一转换为 Tensor 并存储
        self.saved_rewards.append(self._to_tensor(reward))
        self.saved_dones.append(self._to_tensor(done))

    def update(self, next_observation):
        """PPO 学习步：计算 GAE 和 Returns，执行 K_epochs 次优化。"""
        
        # 1. 计算下一个状态的价值 V(s')
        # 如果是 done，next_value 已经是 0
        next_obs_tensor = self._to_tensor(next_observation)
        with torch.no_grad():
            if next_obs_tensor.numel() > 1: # 排除单值 (0) 的情况
                next_value = self.network.get_value(next_obs_tensor).squeeze(-1)
            else: # done 的情况，next_value 为 0
                # 确保 next_value 形状与 rewards/dones 匹配 (例如: (NumEnvs,))
                if self.saved_rewards[0].dim() > 0:
                     next_value = torch.zeros_like(self.saved_rewards[0]).to(self.device)
                else:
                    next_value = torch.tensor(0.0, device=self.device)

        # 2. 准备缓冲区数据 (形状: (Time, NumEnvs, ...))
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
        
        # 使用循环计算 GAE
        for t in reversed(range(num_steps)):
            V_next = next_value if t == num_steps - 1 else old_values[t+1]
            non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * V_next * non_terminal - old_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae

        returns = advantages + old_values
        
        # 4. 扁平化数据 (形状: (Time * NumEnvs, ...))
        b_states = old_states.view(-1, old_states.shape[-1])
        b_actions = old_actions.view(-1, 3)
        b_log_probs = old_log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        
        # 5. 优势归一化
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
        # 6. PPO 梯度更新
        dataset_size = b_states.size(0)
        total_loss, total_entropy, n_updates = 0, 0, 0
        
        for _ in range(self.K_epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
                
                # 提取 Mini-batch
                batch_states = b_states[batch_indices]
                batch_actions = b_actions[batch_indices]
                batch_log_probs = b_log_probs[batch_indices]
                batch_advantages = b_advantages[batch_indices]
                batch_returns = b_returns[batch_indices]
                
                # 前向传播 (新策略)
                base_logits, plant_logits, loc_logits, state_values = self.network(batch_states)
                state_values = state_values.squeeze(-1)
                
                # 计算新策略的 log_prob 和 Entropy
                m_base = Categorical(logits=base_logits)
                m_plant = Categorical(logits=plant_logits)
                m_loc = Categorical(logits=loc_logits)

                a_base = batch_actions[:, 0]
                a_plant = batch_actions[:, 1]
                a_loc = batch_actions[:, 2]

                new_log_prob_base = m_base.log_prob(a_base)
                new_log_prob_plant = m_plant.log_prob(a_plant)
                new_log_prob_loc = m_loc.log_prob(a_loc)

                mask_place = (a_base == 2).float()
                log_probs = new_log_prob_base + mask_place * (new_log_prob_plant + new_log_prob_loc)

                # 熵项
                entropy = m_base.entropy() + mask_place * (m_plant.entropy() + m_loc.entropy())
                
                # PPO Clip Loss
                ratios = torch.exp(log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                
                # 损失函数: Policy Loss (最小化) + 0.5 * Value Loss (MSE) - 0.01 * Entropy Loss (最大化)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(state_values, batch_returns)
                entropy_loss = -0.01 * entropy.mean() # 鼓励探索
                
                loss = policy_loss + value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪 (可选，但推荐用于稳定训练)
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
                
        self.reset_storage()
        return (total_loss / n_updates, total_entropy / n_updates) if n_updates > 0 else (0, 0)

    # ... (save/load 方法不变) ...
    def save(self, nn_name):
        torch.save(self.network.state_dict(), nn_name)

    def load(self, nn_name):
        self.network.load_state_dict(torch.load(nn_name, map_location=self.device))


# 3. 训练器 (Trainer)
class Trainer():
    """环境交互和数据预处理的逻辑封装"""
    def __init__(self, render=False, max_frames=1000):
        # 示例环境创建 (实际使用时需要确保 pvz 环境已注册)
        try:
             self.env = gym.make('gym_pvz:pvz-env-v2')
        except gym.error.UnregisteredEnv:
             # 如果环境未注册，使用一个占位环境
             print("Warning: 'gym_pvz:pvz-env-v2' not found. Using 'CartPole-v1' as a placeholder.")
             self.env = gym.make('CartPole-v1')

        self.max_frames = max_frames
        self.render = render
        
        # 尝试获取原始环境以访问配置
        self.env_base = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def compile_agent_network(self, agent):
        """编译 PPO Agent 中的 ActorCriticNetwork"""
        if hasattr(agent.network, 'shared_trunk') and agent.device.type in ['cuda', 'mps']:
            print(f"Compiling ActorCriticNetwork for performance on {agent.device}...")
            # 使用 torch.compile 进行性能优化
            agent.network = torch.compile(agent.network, fullgraph=True, mode="reduce-overhead")

    def num_observations(self):
        """返回转换后的观察空间维度"""
        # (植物网格) + (僵尸车道总血量) + (阳光值) + (卡牌组信息)
        return self._grid_size + config.N_LANES + len(config.plant_deck) + 1

    def _transform_observation(self, observation):
        """
        对原始观察进行归一化和特征提取。
        原始观察假设为 1D NumPy 数组：[Plant_Grid, Zombie_Grid, Sun, Deck...]
        """
        if hasattr(observation, 'get'):
            observation = observation.get()
        
        observation = np.asarray(observation, dtype=np.float32)
        
        # 1. 植物网格 ([:_grid_size]) - 保持不变
        plant_grid = observation[:self._grid_size]
        
        # 2. 僵尸网格 -> 僵尸血量和（按车道）
        zombie_grid = observation[self._grid_size:2*self._grid_size]
        zombie_grid = zombie_grid.reshape((config.N_LANES, config.LANE_LENGTH))
        # 按车道求和并归一化
        zombie_lane_sum = np.sum(zombie_grid, axis=1) / HP_NORM
        
        # 3. 阳光值 - 归一化
        sun_val = observation[2 * self._grid_size] / SUN_NORM
        
        # 4. 卡牌信息 - 保持不变
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
        进行一集游戏并与 Agent 交互。专为单环境交互设计。
        """
        observation, _info = self.env.reset()
        # 检查是否为 CartPole 占位环境，如果是则跳过 _transform_observation
        if self.env.spec.id.startswith('gym_pvz'):
            observation = self._transform_observation(observation)
        
        episode_steps = 0
        episode_reward = 0.0

        # 简化循环条件
        while episode_steps < self.max_frames:
            if self.render:
                self.env.render()

            # Agent 决定动作
            action = agent.decide_action(observation)
            
            # 环境交互 (observation, reward, terminated, truncated, info)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            
            if self.env.spec.id.startswith('gym_pvz'):
                next_observation = self._transform_observation(next_observation)
            
            # 存储奖励和终止状态 (Agent 负责内部转换)
            agent.store_reward_done(reward, done)

            observation = next_observation
            episode_reward += reward
            episode_steps += 1

            if done:
                break
        
        # PPO 更新需要最后一个状态的价值 V(s')
        if done:
            # 如果终止，V(s') = 0。使用一个标量 0 作为信号
            next_value = np.array(0.0, dtype=np.float32)
        else:
            # 使用最后一个观察值来估算 V(s')
            next_value = observation

        # 触发 PPO 更新
        loss, entropy = agent.update(next_value)
        
        return episode_steps, episode_reward, loss, entropy