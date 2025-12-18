"""
Attention CNN (ACNN) 网络模块

在原始CNN的基础上添加CBAM（Convolutional Block Attention Module）注意力机制：
- Channel Attention: 学习"哪个特征通道更重要"
- Spatial Attention: 学习"哪个空间位置更重要"

论文: CBAM: Convolutional Block Attention Module (ECCV 2018)

状态表示 (与CNN版本相同):
- Channel 0-3: 植物类型嵌入 (5 x 9)
- Channel 4: 僵尸HP网格 (5 x 9)，归一化
- Channel 5: 僵尸存在标记 (5 x 9)，二值化
- Channel 6: 车道威胁等级 (5 x 9)，每车道僵尸总HP广播到整行
- Channel 7: 最近僵尸距离 (5 x 9)，每车道最近僵尸的位置
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


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    
    学习每个特征通道的重要性权重。
    使用全局平均池化 + 全局最大池化双路径，然后通过共享MLP生成通道权重。
    
    输入: (batch, C, H, W)
    输出: (batch, C, 1, 1) 的通道注意力权重
    """
    
    def __init__(self, in_channels, reduction_ratio=4):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: MLP中间层的缩减比例（节省参数）
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP：降维 -> ReLU -> 升维
        reduced_channels = max(in_channels // reduction_ratio, 8)  # 至少8个通道
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, C, H, W) 输入特征图
        
        Returns:
            attention: (batch, C, 1, 1) 通道注意力权重
        """
        batch_size, channels, _, _ = x.size()
        
        # 全局平均池化路径
        avg_out = self.avg_pool(x).view(batch_size, channels)  # (batch, C)
        avg_out = self.mlp(avg_out)  # (batch, C)
        
        # 全局最大池化路径
        max_out = self.max_pool(x).view(batch_size, channels)  # (batch, C)
        max_out = self.mlp(max_out)  # (batch, C)
        
        # 合并两个路径
        attention = torch.sigmoid(avg_out + max_out)  # (batch, C)
        attention = attention.view(batch_size, channels, 1, 1)  # (batch, C, 1, 1)
        
        return attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    学习每个空间位置的重要性权重。
    沿通道维度做平均池化和最大池化，拼接后通过卷积生成空间权重图。
    
    输入: (batch, C, H, W)
    输出: (batch, 1, H, W) 的空间注意力权重
    """
    
    def __init__(self, kernel_size=3):
        """
        Args:
            kernel_size: 卷积核大小（用于聚合周围信息）
                        原论文用7x7，但PvZ网格小(5x9)，用3x3更合适
        """
        super(SpatialAttention, self).__init__()
        
        # 输入是2通道（avg + max），输出是1通道（attention map）
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, C, H, W) 输入特征图
        
        Returns:
            attention: (batch, 1, H, W) 空间注意力权重
        """
        # 沿通道维度做池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, H, W)
        
        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, H, W)
        
        # 卷积生成空间注意力图
        attention = torch.sigmoid(self.conv(concat))  # (batch, 1, H, W)
        
        return attention


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    
    组合通道注意力和空间注意力，顺序应用：
    1. 先应用通道注意力
    2. 再应用空间注意力
    
    支持残差连接，提升训练稳定性。
    
    输入: (batch, C, H, W)
    输出: (batch, C, H, W) 注意力加权后的特征
    """
    
    def __init__(self, in_channels, reduction_ratio=4, spatial_kernel_size=3, 
                 use_residual=True):
        """
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道注意力MLP的缩减比例
            spatial_kernel_size: 空间注意力的卷积核大小
            use_residual: 是否使用残差连接（推荐开启，提升稳定性）
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
        self.use_residual = use_residual
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch, C, H, W) 输入特征图
        
        Returns:
            out: (batch, C, H, W) 注意力加权后的特征
        """
        # 保存输入用于残差连接
        identity = x
        
        # 1. 通道注意力
        channel_attn = self.channel_attention(x)  # (batch, C, 1, 1)
        x = x * channel_attn  # 元素级乘法，广播
        
        # 2. 空间注意力
        spatial_attn = self.spatial_attention(x)  # (batch, 1, H, W)
        x = x * spatial_attn  # 元素级乘法，广播
        
        # 3. 残差连接
        if self.use_residual:
            x = x + identity
        
        return x
    
    def get_attention_maps(self, x):
        """
        获取注意力图（用于可视化）
        
        Returns:
            channel_attn: (batch, C, 1, 1) 通道注意力
            spatial_attn: (batch, 1, H, W) 空间注意力
        """
        channel_attn = self.channel_attention(x)
        x_channel = x * channel_attn
        spatial_attn = self.spatial_attention(x_channel)
        return channel_attn, spatial_attn


class ACNNFeatureExtractor(nn.Module):
    """
    Attention CNN 特征提取器
    
    在原始CNN基础上添加CBAM注意力模块：
    - 在CNN最后一层之后、特征融合之前插入CBAM
    - 让网络学会聚焦于战场中最关键的区域
    
    输入: 原始一维观察向量
    输出: 提取的空间特征向量
    """
    
    def __init__(self, n_plant_types=4, hidden_channels=32, output_features=64,
                 attention_reduction=4, attention_kernel=3, use_residual=True):
        """
        Args:
            n_plant_types: 植物类型数量
            hidden_channels: CNN隐藏通道数
            output_features: 输出特征维度
            attention_reduction: CBAM通道注意力的缩减比例
            attention_kernel: CBAM空间注意力的卷积核大小
            use_residual: CBAM是否使用残差连接
        """
        super(ACNNFeatureExtractor, self).__init__()
        
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
        
        # CBAM注意力模块 - 在CNN最后一层之后
        self.cbam = CBAM(
            in_channels=hidden_channels * 2,
            reduction_ratio=attention_reduction,
            spatial_kernel_size=attention_kernel,
            use_residual=use_residual
        )
        
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
        
        # ===== 车道威胁特征（与CNN版本相同） =====
        # 1. 每车道僵尸总HP（威胁等级）
        lane_threat = zombie_hp_2d.sum(dim=2, keepdim=True)  # (batch, 5, 1)
        lane_threat_normalized = lane_threat / 3.0
        lane_threat_broadcast = lane_threat_normalized.expand(-1, -1, LANE_LENGTH)
        lane_threat_channel = lane_threat_broadcast.unsqueeze(1)  # (batch, 1, 5, 9)
        
        # 2. 每车道最近僵尸的位置（紧急程度）
        zombie_mask = (zombie_hp_2d > 0).float()  # (batch, 5, 9)
        positions = torch.arange(LANE_LENGTH, device=device).view(1, 1, LANE_LENGTH).expand(batch_size, N_LANES, -1)
        positions_masked = torch.where(zombie_mask > 0, positions.float(), torch.tensor(LANE_LENGTH, dtype=torch.float32, device=device))
        nearest_zombie_pos, _ = positions_masked.min(dim=2, keepdim=True)  # (batch, 5, 1)
        nearest_zombie_normalized = nearest_zombie_pos / LANE_LENGTH
        nearest_zombie_broadcast = nearest_zombie_normalized.expand(-1, -1, LANE_LENGTH)
        nearest_zombie_channel = nearest_zombie_broadcast.unsqueeze(1)  # (batch, 1, 5, 9)
        
        # 拼接所有通道 (batch, 8, 5, 9)
        spatial_features = torch.cat([
            plant_features,           # (batch, 4, 5, 9) - 植物类型嵌入
            zombie_hp_normalized,     # (batch, 1, 5, 9) - 僵尸HP
            zombie_presence,          # (batch, 1, 5, 9) - 僵尸存在
            lane_threat_channel,      # (batch, 1, 5, 9) - 车道威胁等级
            nearest_zombie_channel,   # (batch, 1, 5, 9) - 最近僵尸距离
        ], dim=1)
        
        # CNN前向传播
        x = F.leaky_relu(self.conv1(spatial_features))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # ===== CBAM注意力模块 =====
        x = self.cbam(x)
        
        # 展平 (batch, hidden*2*5*9)
        x = x.view(batch_size, -1)
        
        # 车道威胁排名作为全局特征
        urgency = lane_threat_normalized.squeeze(2) * (1.0 - nearest_zombie_normalized.squeeze(2))
        lane_priority = F.softmax(urgency * 5.0, dim=1)
        
        # 全局特征
        global_features = torch.cat([sun, cooldowns, lane_priority], dim=1)
        
        # 融合空间特征和全局特征
        combined = torch.cat([x, global_features], dim=1)
        features = F.leaky_relu(self.fc_fusion(combined))
        
        return features
    
    def forward_with_attention(self, obs):
        """
        前向传播并返回注意力图（用于可视化分析）
        
        Returns:
            features: 提取的特征
            channel_attn: 通道注意力权重
            spatial_attn: 空间注意力图
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # 解析和处理（与forward相同）
        plant_grid = obs[:, :GRID_SIZE].long()
        zombie_hp_grid = obs[:, GRID_SIZE:2*GRID_SIZE]
        sun = obs[:, 2*GRID_SIZE:2*GRID_SIZE+1]
        cooldowns = obs[:, 2*GRID_SIZE+1:]
        
        plant_grid_2d = plant_grid.view(batch_size, N_LANES, LANE_LENGTH)
        zombie_hp_2d = zombie_hp_grid.view(batch_size, N_LANES, LANE_LENGTH)
        
        plant_features = self.plant_embedding(plant_grid_2d)
        plant_features = plant_features.permute(0, 3, 1, 2)
        
        zombie_hp_normalized = zombie_hp_2d.unsqueeze(1)
        zombie_presence = (zombie_hp_2d > 0).float().unsqueeze(1)
        
        lane_threat = zombie_hp_2d.sum(dim=2, keepdim=True)
        lane_threat_normalized = lane_threat / 3.0
        lane_threat_broadcast = lane_threat_normalized.expand(-1, -1, LANE_LENGTH)
        lane_threat_channel = lane_threat_broadcast.unsqueeze(1)
        
        zombie_mask = (zombie_hp_2d > 0).float()
        positions = torch.arange(LANE_LENGTH, device=device).view(1, 1, LANE_LENGTH).expand(batch_size, N_LANES, -1)
        positions_masked = torch.where(zombie_mask > 0, positions.float(), torch.tensor(LANE_LENGTH, dtype=torch.float32, device=device))
        nearest_zombie_pos, _ = positions_masked.min(dim=2, keepdim=True)
        nearest_zombie_normalized = nearest_zombie_pos / LANE_LENGTH
        nearest_zombie_broadcast = nearest_zombie_normalized.expand(-1, -1, LANE_LENGTH)
        nearest_zombie_channel = nearest_zombie_broadcast.unsqueeze(1)
        
        spatial_features = torch.cat([
            plant_features, zombie_hp_normalized, zombie_presence,
            lane_threat_channel, nearest_zombie_channel,
        ], dim=1)
        
        # CNN
        x = F.leaky_relu(self.conv1(spatial_features))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        # 获取注意力图
        channel_attn, spatial_attn = self.cbam.get_attention_maps(x)
        
        # 应用注意力
        x = self.cbam(x)
        
        # 融合
        x = x.view(batch_size, -1)
        urgency = lane_threat_normalized.squeeze(2) * (1.0 - nearest_zombie_normalized.squeeze(2))
        lane_priority = F.softmax(urgency * 5.0, dim=1)
        global_features = torch.cat([sun, cooldowns, lane_priority], dim=1)
        combined = torch.cat([x, global_features], dim=1)
        features = F.leaky_relu(self.fc_fusion(combined))
        
        return features, channel_attn, spatial_attn


class ACNNDuelingQNetwork(nn.Module):
    """
    Attention CNN版本的Dueling Q网络
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    使用ACNN提取带注意力的空间特征，然后分别输出V和A
    """
    
    def __init__(self, env, epsilon=0.05, learning_rate=1e-3, device='cpu',
                 hidden_channels=32, feature_size=128,
                 attention_reduction=4, attention_kernel=3, use_residual=True):
        super(ACNNDuelingQNetwork, self).__init__()
        self.device = device
        
        # 获取植物数量
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        n_plant_types = len(unwrapped_env.plant_deck)
        
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        # ACNN特征提取器（带CBAM注意力）
        self.feature_extractor = ACNNFeatureExtractor(
            n_plant_types=n_plant_types,
            hidden_channels=hidden_channels,
            output_features=feature_size,
            attention_reduction=attention_reduction,
            attention_kernel=attention_kernel,
            use_residual=use_residual
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
                if m.bias is not None:
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
        """Epsilon-greedy动作选择"""
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
        """与evaluate函数兼容的接口"""
        state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.get_action(state, epsilon=epsilon, action_mask=action_mask)
