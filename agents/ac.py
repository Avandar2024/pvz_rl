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

def get_device():
    """自动选择最佳设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 针对 Apple Silicon 的性能优化
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class ActorCriticNetwork(nn.Module):
    """
    Modified Network Structure (Lane Independence & Attention):
    1. Lane Encoder (Shared): CNN + MLP extracting 'Threat/Value' embedding for each lane.
    2. Resource Encoder: Processes global features.
    3. Attention Module: Combines Lane Embeddings and Resource Embedding to score lanes.
    """
    def __init__(self, base_dim=3, plant_dim=4, n_lanes=5, lane_length=9):
        super().__init__()
        self.n_lanes = n_lanes
        self.lane_length = lane_length
        
        # 1. Resource MLP (Global Context)
        self.resource_mlp = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, out_features=64), nn.ReLU()
        )
        
        # 2. Lane CNN + MLP (Local Context)
        self.lane_cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 16 * lane_length
        
        self.lane_mlp = nn.Sequential(
            nn.Linear(64 + cnn_out_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        
        # Heads
        self.global_noop_head = nn.Linear(64, 1)
        self.lane_score_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.actor_plant = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, plant_dim))
        self.actor_location = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, lane_length))
        
        self.critic_head = nn.Sequential(
            nn.Linear(n_lanes * 128, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _extract_features(self, grid, global_features):
        # 1. Global Features
        global_emb = self.resource_mlp(global_features) # (B, 64)
        
        # 2. Lane Features (Shared)
        batch_size = grid.size(0)
        lane_grid = grid.permute(0, 2, 1, 3).reshape(-1, 2, self.lane_length) # (B*N, 2, L)
        
        cnn_feat = self.lane_cnn(lane_grid) # (B*N, cnn_out_dim)
        
        # Expand global embedding to concatenate with each lane's CNN features
        # global_emb: (B, 64) -> (B, N, 64) -> (B*N, 64)
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, self.n_lanes, -1).reshape(-1, 64)
        
        # Concatenate: (B*N, 64 + cnn_out_dim)
        lane_input = torch.cat([global_emb_expanded, cnn_feat], dim=1)
        
        lane_emb = self.lane_mlp(lane_input) # (B*N, 128)
        
        # Reshape back to (B, N, 128)
        lane_emb = lane_emb.view(batch_size, self.n_lanes, -1)
        
        return global_emb, lane_emb, batch_size

    def forward(self, grid, global_features, mask=None):
        global_emb, lane_emb, batch_size = self._extract_features(grid, global_features)
        
        # --- Lane Selection (Attention) ---
        # lane_emb is (B, N, 128)
        
        # Calculate scores
        lane_scores = self.lane_score_head(lane_emb).squeeze(-1) # (B, N)
        noop_score = self.global_noop_head(global_emb) # (B, 1)
        
        base_logits = torch.cat([noop_score, lane_scores], dim=1) # (B, N+1)
        
        # --- Action Heads (Per Lane) ---
        # Input is lane_emb: (B, N, 128)
        
        plant_logits = self.actor_plant(lane_emb) # (B, N, P)
        loc_logits = self.actor_location(lane_emb) # (B, N, L)
        
        if mask is not None:
            mask_reshaped = mask.view(batch_size, self.n_lanes, self.lane_length)
            loc_logits = loc_logits + (mask_reshaped - 1) * 1e9
            
        # --- Critic ---
        # Flatten lane embeddings: (B, N*128)
        lane_emb_flat = lane_emb.view(batch_size, -1)
        # Critic head takes n_lanes * 128
        state_value = self.critic_head(lane_emb_flat)
        
        return base_logits, plant_logits, loc_logits, state_value

    def get_value(self, grid, global_features):
        global_emb, lane_emb, batch_size = self._extract_features(grid, global_features)
        lane_emb_flat = lane_emb.view(batch_size, -1)
        return self.critic_head(lane_emb_flat)

# 2. PPO Agent (策略梯度算法)
class PPOAgent():
    """Proximal Policy Optimization (PPO) Agent 的实现。"""
    def __init__(self, lr=1e-4, gamma=0.99, eps_clip=0.2, K_epochs=20, gae_lambda=0.95, mini_batch_size=256, entropy_coef=0.03, reward_scale=100.0, possible_actions=None):
        
        self.device = get_device()
        print(f"Using device: {self.device}")

        # 固定的输出维度
        self.base_dim = 3
        self.plant_dim = 4
        self.location_dim = config.N_LANES * config.LANE_LENGTH
        self.n_lanes = config.N_LANES
        self.lane_length = config.LANE_LENGTH

        self.network = ActorCriticNetwork(self.base_dim, self.plant_dim,  self.n_lanes, self.lane_length).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        # 动态学习率衰减
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.posible_actions = possible_actions
        
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
        """统一的数据转换和设备移动函数，处理 np.ndarray, CuPy, Tensor

        Ensures tensors moved to the agent device with a safe dtype (float32
        on MPS), avoiding float64 which MPS doesn't support.
        """
        # If user requested float64 on MPS, override to float32 to avoid errors
        if self.device.type == 'mps' and dtype == torch.float64:
            dtype = torch.float32

        # Use torch.as_tensor for numpy / scalars to control dtype+device in one step
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=dtype)
        else:
            tensor = torch.as_tensor(data, dtype=dtype, device=self.device)

        if is_action:
            return tensor.long()
        return tensor

    def _split_state(self, state_tensor):
        """Split flattened state into grid, global features and mask."""
        grid_size = self.n_lanes * self.lane_length
        grid_end = 2 * grid_size
        
        # Mask is at the end (size: grid_size)
        mask = state_tensor[:, -grid_size:]
        
        # Global features is between grid and mask
        global_features = state_tensor[:, grid_end:-grid_size]
        
        grid_flat = state_tensor[:, :grid_end]
        
        batch_size = state_tensor.size(0)
        grid = grid_flat.view(batch_size, 2, self.n_lanes, self.lane_length)
        return grid, global_features, mask

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
            
        grid, global_features, mask = self._split_state(state_tensor)
        base_logits, plant_logits, loc_logits, value = self.network(grid, global_features, mask)
        value = value.squeeze(-1) # (B, 1) -> (B,)
        
        # 1. Select Base Action (No-op vs Which Lane?)
        # base_logits: (B, N+1)
        m_base = Categorical(logits=base_logits)
        action_base = m_base.sample() # (B,) values in 0..N
        log_prob_base = m_base.log_prob(action_base)

        # 2. Select Plant & Location for ALL lanes (for batch efficiency)
        # plant_logits: (B, N, P)
        m_plant = Categorical(logits=plant_logits)
        action_plant = m_plant.sample() # (B, N)
        log_prob_plant = m_plant.log_prob(action_plant) # (B, N)

        # loc_logits: (B, N, L)
        m_loc = Categorical(logits=loc_logits)
        action_loc = m_loc.sample() # (B, N)
        log_prob_loc = m_loc.log_prob(action_loc) # (B, N)

        # Create mask for placement (B,)
        is_placement = (action_base > 0)
        
        # Get lane index (0..N-1) for placement actions. 
        # Use clamp to avoid index out of bounds for No-op (index -1), though we'll mask it out.
        lane_indices = (action_base - 1).clamp(min=0)
        
        # Gather log probs for the selected lanes
        # log_prob_plant: (B, N) -> gather -> (B,)
        selected_log_prob_plant = log_prob_plant.gather(1, lane_indices.unsqueeze(1)).squeeze(1)
        selected_log_prob_loc = log_prob_loc.gather(1, lane_indices.unsqueeze(1)).squeeze(1)
        
        combined_log_prob = log_prob_base + is_placement.float() * (selected_log_prob_plant + selected_log_prob_loc)
        
        stored_actions = torch.cat([
            action_base.unsqueeze(1), 
            action_plant, 
            action_loc
        ], dim=1) # (B, 1 + 2N)

        # 存储经验。如果是单环境，需要存储 (D,) 的 Tensor
        stored_states = state_tensor.detach().clone()
        stored_log_probs = combined_log_prob.detach().clone()
        stored_values = value.detach().clone()

        self.saved_states.append(stored_states if not is_single_env else stored_states[0])
        self.saved_actions.append(stored_actions if not is_single_env else stored_actions[0])
        self.saved_log_probs.append(stored_log_probs if not is_single_env else stored_log_probs[0])
        self.saved_values.append(stored_values if not is_single_env else stored_values[0])

        # 直接在 Tensor 层面计算环境所需的标量动作，避免 Python 循环和 map_action 导致计算图断裂
        selected_plant = action_plant.gather(1, lane_indices.unsqueeze(1)).squeeze(1)
        selected_loc = action_loc.gather(1, lane_indices.unsqueeze(1)).squeeze(1)
        
        env_action = 1 + (selected_loc * self.n_lanes + lane_indices) * self.plant_dim + selected_plant
        env_action = torch.where(action_base == 0, torch.tensor(0, device=self.device), env_action)

        env_action_np = env_action.cpu().numpy()

        if is_single_env:
            return env_action_np.item()
        else:
            return env_action_np
        
    def store_reward_done(self, reward, done):
        """存储奖励和终止信号 (支持单环境或向量化环境)"""
        # 统一转换为 Tensor 并存储
        # 对 reward 进行缩放归一化，防止高分时 Value Loss 爆炸
        if isinstance(reward, np.ndarray):
            scaled_reward = reward / self.reward_scale
        else:
            scaled_reward = reward / self.reward_scale
            
        self.saved_rewards.append(self._to_tensor(scaled_reward))
        self.saved_dones.append(self._to_tensor(done))

    def update(self, next_observation):
        """PPO 学习步：计算 GAE 和 Returns，执行 K_epochs 次优化。"""
        
        # 1. 计算下一个状态的价值 V(s')
        # 如果是 done，next_value 已经是 0
        next_obs_tensor = self._to_tensor(next_observation)
        with torch.no_grad():
            if next_obs_tensor.numel() > 1: # 排除单值 (0) 的情况
                if next_obs_tensor.dim() == 1:
                    next_obs_tensor = next_obs_tensor.unsqueeze(0)
                grid, global_features, _ = self._split_state(next_obs_tensor)
                next_value = self.network.get_value(grid, global_features).squeeze(-1)
                if next_value.numel() == 1:
                    next_value = next_value.squeeze()
            else: # done 的情况，next_value 为 0
                # 确保 next_value 形状与 rewards/dones 匹配 (例如: (NumEnvs,))
                if self.saved_rewards[0].dim() > 0:
                     next_value = torch.zeros_like(self.saved_rewards[0]).to(self.device)
                else:
                    next_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)

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
        # Action shape is now (1 + 2N)
        b_actions = old_actions.view(-1, old_actions.shape[-1])
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
                grid, global_features, mask = self._split_state(batch_states)
                base_logits, plant_logits, loc_logits, state_values = self.network(grid, global_features, mask)
                state_values = state_values.squeeze(-1)
                
                # 计算新策略的 log_prob 和 Entropy
                m_base = Categorical(logits=base_logits)
                m_plant = Categorical(logits=plant_logits)
                m_loc = Categorical(logits=loc_logits)

                a_base = batch_actions[:, 0]
                a_plant = batch_actions[:, 1 : 1 + self.n_lanes]
                a_loc = batch_actions[:, 1 + self.n_lanes :]

                new_log_prob_base = m_base.log_prob(a_base)
                
                # Calculate log probs for ALL lanes
                # m_plant.log_prob(a_plant) -> (B, N)
                new_log_prob_plant = m_plant.log_prob(a_plant)
                new_log_prob_loc = m_loc.log_prob(a_loc)

                # Select the relevant log probs based on a_base
                is_placement = (a_base > 0)
                lane_indices = (a_base - 1).clamp(min=0).long()
                
                selected_log_prob_plant = new_log_prob_plant.gather(1, lane_indices.unsqueeze(1)).squeeze(1)
                selected_log_prob_loc = new_log_prob_loc.gather(1, lane_indices.unsqueeze(1)).squeeze(1)

                log_probs = new_log_prob_base + is_placement.float() * (selected_log_prob_plant + selected_log_prob_loc)

                # 熵项
                # Entropy = H(Base) + sum(P(Lane k) * (H(Plant|k) + H(Loc|k)))
                # P(Lane k) is softmax(base_logits)[k+1]
                probs_base = F.softmax(base_logits, dim=1) # (B, N+1)
                
                entropy_base = m_base.entropy()
                entropy_plant = m_plant.entropy() # (B, N)
                entropy_loc = m_loc.entropy() # (B, N)
             
                # probs_base[:, 1:] corresponds to lanes 0..N-1
                lane_probs = probs_base[:, 1:] # (B, N)
                weighted_sub_entropy = (lane_probs * (entropy_plant + entropy_loc)).sum(dim=1)
                
                entropy = entropy_base + weighted_sub_entropy
                
                # PPO Clip Loss
                ratios = torch.exp(log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                
                # 损失函数: Policy Loss (最小化) + 0.5 * Value Loss (Huber) - entropy_coef * Entropy Loss (最大化)
                policy_loss = -torch.min(surr1, surr2).mean()
                # print(f"policy_loss: {policy_loss.item():.4f}")
                # 使用 Smooth L1 Loss (Huber Loss) 替代 MSE，防止 Value Loss 爆炸导致梯度被 Value 支配
                value_loss = 0.5 * F.smooth_l1_loss(state_values, batch_returns)
                # print(f"value_loss: {value_loss.item():.4f}")
                entropy_loss = -self.entropy_coef * entropy.mean() # 鼓励探索
                
                loss = policy_loss + value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
                
        self.scheduler.step()
        self.reset_storage()
        return (total_loss / n_updates, total_entropy / n_updates) if n_updates > 0 else (0, 0)

    def save(self, nn_name):
        torch.save(self.network.state_dict(), nn_name)

    def load(self, nn_name):
        state_dict = torch.load(nn_name, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        self.network.load_state_dict(new_state_dict)


# 3. 训练器 (Trainer)
class Trainer():
    """环境交互和数据预处理的逻辑封装"""
    def __init__(self, render=False, max_frames=1000, training=True):
        self.env = gym.make('gym_pvz:pvz-env-v3')

        self.max_frames = max_frames
        self.render = render
        self.training = training
        
        # 尝试获取原始环境以访问配置
        self.env_base = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        self._grid_size = config.N_LANES * config.LANE_LENGTH

    def compile_agent_network(self, agent):
        """编译 PPO Agent 中的 ActorCriticNetwork"""
        if agent.device.type in ['cuda', 'mps']:
            print(f"Compiling ActorCriticNetwork for performance on {agent.device}...")
            # 使用 torch.compile 进行性能优化
            agent.network = torch.compile(agent.network, fullgraph=True, mode="reduce-overhead")

    def num_actions(self):
        """返回环境的离散动作数。"""
        return int(self.env.action_space.n)

    def _transform_observation(self, observation):
        """
        对原始观察进行归一化和特征提取。
        支持单个观察 (N,) 或批量观察 (B, N)。
        """
        observation = np.asarray(observation, dtype=np.float32)
        
        # 记录是否为单个输入
        is_single = observation.ndim == 1
        if is_single:
            observation = observation[None, :]
            
        # 统一按批量处理 (B, Features)
        # 1. 植物网格
        plant_grid = observation[:, :self._grid_size]

        # 2. 僵尸网格 -> 归一化
        zombie_grid = observation[:, self._grid_size:2*self._grid_size] / HP_NORM

        # 3. 阳光值 - 归一化 (保持维度为 (B,1))
        sun_val = observation[:, 2 * self._grid_size:2 * self._grid_size + 1] / SUN_NORM

        # Calculate n_plants dynamically
        total_len = observation.shape[-1]
        # Structure: plant_grid(G) + zombie_grid(G) + sun(1) + card_info(N) + mask(G)
        # total = 2G + 1 + N + G = 3G + 1 + N
        # N = total - 3G - 1
        n_plants = total_len - 3 * self._grid_size - 1

        # 4. 卡牌信息
        card_info = observation[:, 2 * self._grid_size+1 : 2 * self._grid_size+1 + n_plants]
        
        # 5. 位置掩码
        location_mask = observation[:, 2 * self._grid_size+1 + n_plants:]

        # 拼接 (B, NewFeatures)
        new_observation = np.concatenate([
            plant_grid,
            zombie_grid,
            sun_val,
            card_info,
            location_mask
        ], axis=1).astype(np.float32)

        if is_single:
            return new_observation[0]
            
        return new_observation

    def _run_episode(self, agent):
        observation, _ = self.env.reset()
        observation = self._transform_observation(observation)
        
        episode_steps = 0
        episode_reward = 0.0
        actions = []
        rewards = []
        
        while episode_steps < self.max_frames:
            if self.render:
                self.env.render()

            action = agent.decide_action(observation)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            done = terminated or truncated
            next_obs = self._transform_observation(next_obs)
            
            agent.store_reward_done(reward, done)
            
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            episode_steps += 1
            
            observation = next_obs
            
            if done:
                break
                
        return observation, done, actions, rewards, episode_steps, episode_reward

    def play(self, agent):
        """
        进行一集游戏并与 Agent 交互。专为单环境交互设计。
        """
        last_obs, done, actions, rewards, steps, ep_reward = self._run_episode(agent)
        
        # PPO 更新需要最后一个状态的价值 V(s')
        loss, entropy = 0, 0
        if self.training:
            # 如果终止，V(s') = 0。使用一个标量 0 作为信号
            next_value = np.array(0.0, dtype=np.float32) if done else last_obs
            loss, entropy = agent.update(next_value)
        else:
            agent.reset_storage()
        
        return {
            "rewards": np.array(rewards),
            "actions": np.array(actions),
            "episode_steps": steps,
            "episode_reward": ep_reward,
            "loss": loss,
            "entropy": entropy
        }

    def close(self):
        """释放底层环境资源。"""
        self.env.close()