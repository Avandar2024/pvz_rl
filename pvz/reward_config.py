"""
奖励配置文件

集中管理所有奖励相关的参数，包括：
- 终局奖励（胜利/失败/超时）
- 即时奖励（击杀/生命损失/存活）
- 奖励塑形（位置评估、种植引导）
"""

# ==================== 终局奖励 ====================
REWARD_WIN = 1000.0            # 胜利奖励
REWARD_LOSE = -800.0           # 失败惩罚
REWARD_TIMEOUT = -500.0        # 超时惩罚

# ==================== 即时奖励 ====================
# 击杀奖励
ZOMBIE_KILL_REWARD = 40.0     # 击杀普通僵尸奖励 (原 Zombie.SCORE 120)
ZOMBIE_CONE_KILL_REWARD = 60.0   # 击杀路障僵尸奖励
ZOMBIE_BUCKET_KILL_REWARD = 80.0 # 击杀铁桶僵尸奖励
ZOMBIE_FLAG_KILL_REWARD = 40.0   # 击杀旗帜僵尸奖励

# 惩罚
LIFE_LOSS_PENALTY = -100.0     # 僵尸进家惩罚（每损失一条生命）

# 存活奖励
SURVIVAL_REWARD = 0.0          # 每个存活间隔的奖励 (原 config.SURVIVAL)
SURVIVAL_STEP = 20.0           # 存活奖励间隔（秒）(原 config.SURVIVAL_STEP)
PLANT_ALIVE_REWARD = 0.0       # 每帧每株存活植物奖励 (原 config.SCORE_ALIVE_PLANT)
MOWER_ALIVE_REWARD = 0.0       # 每帧每个存活割草机奖励 (原 config.SCORE_ALIVE_MOWER)

# ==================== 局面评估参数 ====================
# 分数范围设计
LANE_SCORE_MAX = 150          # 单行最高分
ECONOMY_SCORE_MAX = 150       # 经济最高分
TOTAL_SCORE_MAX = 1500        # 总分上限

# 奖励塑形系数
SHAPING_COEFFICIENT = 1.2    # 局面评估差分奖励的缩放系数

# ==================== 危险区域惩罚 ====================
# 僵尸进入危险区域（靠近房子）时的持续惩罚
DANGER_ZONE_THRESHOLD = 5      # 危险区域阈值：0-4列为危险区域
DANGER_ZONE_PENALTY_BASE = -1.5 # 基础惩罚（每僵尸每次检测）
DANGER_ZONE_PENALTY_SCALE = 2.0  # 越靠近左边惩罚越大的缩放系数
# 惩罚公式: penalty = DANGER_ZONE_PENALTY_BASE * (DANGER_ZONE_THRESHOLD - zombie_col) * DANGER_ZONE_PENALTY_SCALE

# ==================== 种植位置奖励 ====================
# 引导智能体学习合理的植物布局
PLACEMENT_REWARDS = {
    'sunflower': {
        0: 15,    # 最左列 (最佳，安全且不占防御位)
        1: 0,     # 第二列 (可接受)
    },
    'peashooter': {
        1: 10,    # 第二列 (最佳，攻击范围好)
        2: 5,     # 第三列 (次佳)
        3: 3,     # 第四列 (可接受)
    },
    'wallnut': {
        3: 5,
        4: 10,
        5: 5,
        6: 2,
    },
    'potatomine': {
        2: 2,
        3: 2,
        4: 2,
        5: 2,
    }
}

# 默认奖励（未在字典中指定的列）
PLACEMENT_DEFAULT = {
    'sunflower': -5,
    'peashooter': 0,
    'wallnut': 0,
    'potatomine': 0,
}

# 第一排（最左列）种植非向日葵的额外惩罚
FRONT_ROW_PENALTY = -10.0

# ==================== 阵型组合奖励 ====================
# 评估植物之间的协同配合
COMBO_WALLNUT_PROTECTS_SHOOTER = 20.0   # 坚果保护豌豆射手的组合奖励
COMBO_WALLNUT_BLOCKS_ZOMBIE = 12.0      # 坚果阻挡僵尸且左侧有空间放输出的奖励
COMBO_POTATO_BEHIND_WALLNUT = 8.0       # 土豆雷在坚果左侧的组合奖励