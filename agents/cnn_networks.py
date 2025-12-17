"""
CNN网络模块

将PvZ的状态表示为2D网格，使用CNN提取空间特征。

状态表示 (增强版):
- Channel 0-3: 植物类型嵌入 (5 x 9)
- Channel 4: 僵尸HP网格 (5 x 9)，归一化
- Channel 5: 僵尸存在标记 (5 x 9)，二值化
- Channel 6: 车道威胁等级 (5 x 9)，每车道僵尸总HP广播到整行
- Channel 7: 最近僵尸距离 (5 x 9)，每车道最近僵尸的位置

全局特征（拼接到CNN输出后）:
- 阳光值 (归一化)
- 植物冷却状态 (4个)
- 车道威胁排名 (5个，标识最危险的车道)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pvz import config


# 游戏参数
N_LANES = config.N_LANES  # 5
LANE_LENGTH = config.LANE_LENGTH  # 9
GRID_SIZE = N_LANES * LANE_LENGTH  # 45
MAX_ZOMBIE_HP = 1000  # 用于归一化


class CNNFeatureExtractor(nn.Module):
    """
    CNN特征提取器 (增强版)
    
    新增车道威胁特征，帮助模型学习"哪条车道最需要防守"
    
    输入: 原始一维观察向量
    输出: 提取的空间特征向量
    """
    
    def __init__(self, n_plant_types=4, hidden_channels=32, output_features=64):
        super(CNNFeatureExtractor, self).__init__()
        
        self.n_plant_types = n_plant_types
        self.hidden_channels = hidden_channels
        self.output_features = output_features
        
        # 植物类型嵌入（将离散类型转为连续向量）
        # 0=空, 1-4=四种植物
        self.plant_embedding = nn.Embedding(n_plant_types + 1, 4)
        
        # CNN层 - 处理 (batch, channels, 5, 9) 的输入
        # 输入通道: 4(植物嵌入) + 1(僵尸HP) + 1(僵尸存在) + 1(车道威胁) + 1(最近僵尸距离) = 8
        n_input_channels = 8
        self.conv1 = nn.Conv2d(n_input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1)
        
        # 计算卷积后的特征大小
        # 5x9 -> 5x9 (padding=1保持大小)
        conv_output_size = hidden_channels * 2 * N_LANES * LANE_LENGTH
        
        # 全局特征: 阳光(1) + 冷却状态(n_plant_types) + 车道威胁排名(5)
        global_feature_size = 1 + n_plant_types + N_LANES
        
        # 特征融合层
        self.fc_fusion = nn.Linear(conv_output_size + global_feature_size, output_features)
        
    def forward(self, obs):
        """
        前向传播
        
        Args:
            obs: (batch, obs_dim) 原始观察向量
                 obs_dim = 45(植物) + 45(僵尸HP) + 1(阳光) + 4(冷却)
        
        Returns:
            features: (batch, output_features) 提取的特征
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # 解析观察向量
        plant_grid = obs[:, :GRID_SIZE].long()  # (batch, 45) 植物类型
        zombie_hp_grid = obs[:, GRID_SIZE:2*GRID_SIZE]  # (batch, 45) 僵尸HP
        sun = obs[:, 2*GRID_SIZE:2*GRID_SIZE+1]  # (batch, 1) 阳光
        cooldowns = obs[:, 2*GRID_SIZE+1:]  # (batch, 4) 冷却状态
        
        # 重塑为2D网格 (batch, 5, 9)
        plant_grid_2d = plant_grid.view(batch_size, N_LANES, LANE_LENGTH)
        zombie_hp_2d = zombie_hp_grid.view(batch_size, N_LANES, LANE_LENGTH)
        
        # 植物嵌入 (batch, 5, 9) -> (batch, 5, 9, 4) -> (batch, 4, 5, 9)
        plant_features = self.plant_embedding(plant_grid_2d)  # (batch, 5, 9, 4)
        plant_features = plant_features.permute(0, 3, 1, 2)  # (batch, 4, 5, 9)
        
        # 僵尸基础特征
        zombie_hp_normalized = zombie_hp_2d.unsqueeze(1)  # (batch, 1, 5, 9)
        zombie_presence = (zombie_hp_2d > 0).float().unsqueeze(1)  # (batch, 1, 5, 9)
        
        # ===== 新增：车道威胁特征 =====
        # 1. 每车道僵尸总HP（威胁等级）
        lane_threat = zombie_hp_2d.sum(dim=2, keepdim=True)  # (batch, 5, 1)
        # 归一化威胁值 (假设最大威胁约3000HP，即3只铁桶僵尸)
        lane_threat_normalized = lane_threat / 3.0  # 已经除过1000了，再除3
        # 广播到整行 (batch, 5, 9)
        lane_threat_broadcast = lane_threat_normalized.expand(-1, -1, LANE_LENGTH)
        lane_threat_channel = lane_threat_broadcast.unsqueeze(1)  # (batch, 1, 5, 9)
        
        # 2. 每车道最近僵尸的位置（紧急程度）
        # 找到每车道最左边的僵尸位置
        zombie_mask = (zombie_hp_2d > 0).float()  # (batch, 5, 9)
        # 位置索引 0-8，乘以存在掩码，不存在的设为9（最远）
        positions = torch.arange(LANE_LENGTH, device=device).view(1, 1, LANE_LENGTH).expand(batch_size, N_LANES, -1)
        # 对于没有僵尸的位置，设为一个大值
        positions_masked = torch.where(zombie_mask > 0, positions.float(), torch.tensor(LANE_LENGTH, dtype=torch.float32, device=device))
        # 每车道最近僵尸位置
        nearest_zombie_pos, _ = positions_masked.min(dim=2, keepdim=True)  # (batch, 5, 1)
        # 归一化到0-1（0=在家门口，1=在最右边或没有僵尸）
        nearest_zombie_normalized = nearest_zombie_pos / LANE_LENGTH
        # 广播到整行
        nearest_zombie_broadcast = nearest_zombie_normalized.expand(-1, -1, LANE_LENGTH)
        nearest_zombie_channel = nearest_zombie_broadcast.unsqueeze(1)  # (batch, 1, 5, 9)
        
        # 拼接所有通道 (batch, 8, 5, 9)
        spatial_features = torch.cat([
            plant_features,           # (batch, 4, 5, 9) - 植物类型嵌入
            zombie_hp_normalized,     # (batch, 1, 5, 9) - 僵尸HP
            zombie_presence,          # (batch, 1, 5, 9) - 僵尸存在
            lane_threat_channel,      # (batch, 1, 5, 9) - 车道威胁等级 [新增]
            nearest_zombie_channel,   # (batch, 1, 5, 9) - 最近僵尸距离 [新增]
        ], dim=1)
        
        # CNN前向传播
        x = F.leaky_relu(self.conv1(spatial_features))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # 展平 (batch, hidden*2*5*9)
        x = x.view(batch_size, -1)
        
        # ===== 新增：车道威胁排名作为全局特征 =====
        # 计算每车道的"紧急程度" = 威胁 * (1 - 距离)，距离越近越紧急
        urgency = lane_threat_normalized.squeeze(2) * (1.0 - nearest_zombie_normalized.squeeze(2))  # (batch, 5)
        # softmax得到概率分布，表示"应该优先防守哪条车道"
        lane_priority = F.softmax(urgency * 5.0, dim=1)  # (batch, 5)，乘5增加区分度
        
        # 全局特征
        global_features = torch.cat([sun, cooldowns, lane_priority], dim=1)  # (batch, 5 + 5 = 10)
        
        # 融合空间特征和全局特征
        combined = torch.cat([x, global_features], dim=1)
        features = F.leaky_relu(self.fc_fusion(combined))
        
        return features


class CNNQNetwork(nn.Module):
    """
    CNN版本的Q网络 (用于DDQN)
    
    使用CNN提取空间特征，然后用全连接层输出Q值
    """
    
    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu',
                 hidden_channels=32, feature_size=128):
        super(CNNQNetwork, self).__init__()
        self.device = device
        
        # 获取植物数量
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        n_plant_types = len(unwrapped_env.plant_deck)
        
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # CNN特征提取器
        self.feature_extractor = CNNFeatureExtractor(
            n_plant_types=n_plant_types,
            hidden_channels=hidden_channels,
            output_features=feature_size
        )
        
        # Q值输出头
        self.q_head = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.LeakyReLU(),
            nn.Linear(feature_size, self.n_outputs)
        )
        
        # 初始化
        self._initialize_weights()
        self.to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Epsilon参数
        self.epsilon = epsilon
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播，输出所有动作的Q值"""
        features = self.feature_extractor(x)
        q_values = self.q_head(features)
        return q_values
    
    def greedy_action(self, state):
        """贪婪动作选择"""
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1)
    
    def get_action(self, state, epsilon=None, action_mask=None):
        """
        Epsilon-greedy动作选择
        
        Args:
            state: 状态张量
            epsilon: 探索率（None则使用self.epsilon）
            action_mask: 可用动作掩码
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            # 随机探索
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            return np.random.choice(self.actions)
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(state)
                if action_mask is not None:
                    # 将不可用动作的Q值设为负无穷
                    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                    q_values[~mask_tensor.unsqueeze(0)] = float('-inf')
                return torch.argmax(q_values, dim=1).item()
    
    def decide_action(self, observation, action_mask, epsilon=0):
        """
        与evaluate函数兼容的接口
        
        Args:
            observation: numpy观察向量
            action_mask: 可用动作掩码
            epsilon: 探索率
        """
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.get_action(state, epsilon=epsilon, action_mask=action_mask)


class CNNDuelingQNetwork(nn.Module):
    """
    CNN版本的Dueling Q网络 (用于D3QN/DDDQN)
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    使用CNN提取空间特征，然后分别输出V和A
    """
    
    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu',
                 hidden_channels=32, feature_size=128):
        super(CNNDuelingQNetwork, self).__init__()
        self.device = device
        
        # 获取植物数量
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        n_plant_types = len(unwrapped_env.plant_deck)
        
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # CNN特征提取器
        self.feature_extractor = CNNFeatureExtractor(
            n_plant_types=n_plant_types,
            hidden_channels=hidden_channels,
            output_features=feature_size
        )
        
        # Value流 - 输出状态价值 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_size // 2, 1)
        )
        
        # Advantage流 - 输出动作优势 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.LeakyReLU(),
            nn.Linear(feature_size // 2, self.n_outputs)
        )
        
        # 初始化
        self._initialize_weights()
        self.to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Epsilon参数
        self.epsilon = epsilon
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        """
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        
        # Dueling组合: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def greedy_action(self, state):
        """贪婪动作选择"""
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1)
    
    def get_action(self, state, epsilon=None, action_mask=None):
        """
        Epsilon-greedy动作选择
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return np.random.choice(valid_actions)
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                if action_mask is not None:
                    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=self.device)
                    q_values[~mask_tensor.unsqueeze(0)] = float('-inf')
                return torch.argmax(q_values, dim=1).item()
    
    def decide_action(self, observation, action_mask, epsilon=0):
        """
        与evaluate函数兼容的接口
        """
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.get_action(state, epsilon=epsilon, action_mask=action_mask)


# ==================== Player类（用于评估） ====================

# 归一化常量
HP_NORM = 1000
SUN_NORM = 1000


class PlayerCNN:
    """
    CNN版本的Player，用于evaluate函数
    
    与原版PlayerQ接口兼容，但不需要做额外的transform（CNN内部处理）
    """
    
    def __init__(self, env=None, render=False):
        if env is None:
            import gymnasium as gym
            self.env = gym.make('gym_pvz:pvz-env-v3')
        else:
            self.env = env
        self.render_mode = render
        self._grid_size = N_LANES * LANE_LENGTH
    
    def get_actions(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return list(range(env.action_space.n))
    
    def num_observations(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return 2 * N_LANES * LANE_LENGTH + 1 + len(env.plant_deck)
    
    def num_actions(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env.action_space.n
    
    def _transform_observation(self, observation):
        """保持原始观察，CNN内部会处理归一化"""
        return observation.astype(np.float64)
    
    def play(self, agent, epsilon=0):
        """
        玩一局游戏，收集观察和奖励
        
        Args:
            agent: 网络模型（CNNQNetwork或CNNDuelingQNetwork）
            epsilon: 探索率
        """
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
            if self.render_mode:
                self.env.render()
            
            # 获取动作掩码
            env_for_mask = self.env
            while hasattr(env_for_mask, 'env'):
                env_for_mask = env_for_mask.env
            try:
                mask = env_for_mask.mask_available_actions()
            except Exception:
                mask = np.full(self.num_actions(), True)
            
            # 选择动作（调用网络的get_action方法）
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
