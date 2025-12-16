"""
奖励配置文件

集中管理所有奖励相关的参数，包括：
- 终局奖励（胜利/失败/超时）
- 即时奖励（击杀/生命损失/存活）
- 奖励塑形（位置评估、种植引导）
"""

# ==================== 终局奖励 ====================
REWARD_WIN = 500.0            # 胜利奖励
REWARD_LOSE = -1000.0           # 失败惩罚
REWARD_TIMEOUT = -300.0        # 超时惩罚

# ==================== 即时奖励 ====================
# 击杀奖励
ZOMBIE_KILL_REWARD = 72.0     # 击杀普通僵尸奖励 (原 Zombie.SCORE 120)
ZOMBIE_CONE_KILL_REWARD = 72.0   # 击杀路障僵尸奖励
ZOMBIE_BUCKET_KILL_REWARD = 72.0 # 击杀铁桶僵尸奖励
ZOMBIE_FLAG_KILL_REWARD = 72.0   # 击杀旗帜僵尸奖励

# 惩罚
LIFE_LOSS_PENALTY = -200.0     # 僵尸进家惩罚（每损失一条生命）

# 存活奖励
SURVIVAL_REWARD = 0.0          # 每个存活间隔的奖励 (原 config.SURVIVAL)
SURVIVAL_STEP = 20.0           # 存活奖励间隔（秒）(原 config.SURVIVAL_STEP)
PLANT_ALIVE_REWARD = 0.0       # 每帧每株存活植物奖励 (原 config.SCORE_ALIVE_PLANT)
MOWER_ALIVE_REWARD = 0.0       # 每帧每个存活割草机奖励 (原 config.SCORE_ALIVE_MOWER)

# ==================== 局面评估参数 ====================
# 分数范围设计
LANE_SCORE_MAX = 150          # 单行最高分
ECONOMY_SCORE_MAX = 150       # 经济最高分
TOTAL_SCORE_MAX = 1000        # 总分上限

# 奖励塑形系数
SHAPING_COEFFICIENT = 0.5    # 局面评估差分奖励的缩放系数

# ==================== 种植位置奖励 ====================
# 引导智能体学习合理的植物布局
PLACEMENT_REWARDS = {
    'sunflower': {
        0: 30,    # 最左列 (最佳，安全且不占防御位)
        1: 5,     # 第二列 (可接受)
        2: -5,    # 第三列 (不推荐，占用防御位)
    },
    'peashooter': {
        1: 15,    # 第二列 (最佳，攻击范围好)
        2: 10,    # 第三列 (次佳)
        3: 5,     # 第四列 (可接受)
        0: -5,    # 最左列 (太靠前，容易被吃)
    },
    'wallnut': {
        4: 8,     # 中间偏右 (拖延僵尸)
        5: 10,    # 较理想 (拖延+保护后排)
        6: 8,
    },
    'potatomine': {
        0: 0,
    }
}

# 默认奖励（未在字典中指定的列）
PLACEMENT_DEFAULT = {
    'sunflower': -10,   # 其他位置都不推荐
    'peashooter': 0,    # 中性
    'wallnut': 0,       # 中性
    'potatomine': 0,    # 中性
}

# 阳光浪费惩罚：在没有僵尸的行种植非向日葵植物
SUN_WASTE_PENALTY = -10.0

# 第一排（最左列）种植非向日葵的额外惩罚
FRONT_ROW_PENALTY = -15.0