# ==================== 游戏基础参数 ====================
FPS = 2                     # 游戏帧率
MAX_FRAMES = 400            # 最大游戏帧数 (超过则胜利)

N_LANES = 5                 # 行数 (高度)
LANE_LENGTH = 9             # 列数 (宽度)

INITIAL_SUN_AMOUNT = 50     # 初始阳光数量

# 天然阳光产出
NATURAL_SUN_PRODUCTION = 25           # 每次产出阳光数量
NATURAL_SUN_PRODUCTION_COOLDOWN = 10  # 产出间隔 (秒)

# 割草机
MOWERS = False              # 是否启用割草机

# 原始得分参数 (Scene.score 使用)
SURVIVAL = 0                # 存活奖励基础值
SURVIVAL_STEP = 20          # 存活奖励增加间隔 (秒)
SCORE_ALIVE_PLANT = 0       # 每帧每株存活植物得分
SCORE_ALIVE_MOWER = 0       # 每帧每个存活割草机得分


# ==================== 奖励塑形参数 (PBRS) ====================
# 基于潜力函数的奖励塑形 (Potential-Based Reward Shaping)
# 总奖励 = 基础奖励 + 稀疏奖励 + PBRS奖励 + 存活奖励
# PBRS奖励 F = γ * Φ(s') - Φ(s)
# 潜力函数 Φ(s) = w_sun*Φ_sun + w_defense*Φ_defense + w_threat*Φ_threat + w_kills*Φ_kills

# --- 稀疏奖励 (关键节点的大奖励/惩罚) ---
REWARD_WIN = 1000.0         # 通关胜利奖励 (撑过 MAX_FRAMES)
REWARD_LOSE = -1000.0       # 失败惩罚 (僵尸进入房子) [与胜利对称，强调最终目标]

# --- 密集奖励 ---
REWARD_SURVIVAL_PER_FRAME = 0.1  # 每帧存活的小奖励 (鼓励活得更久)

# --- PBRS 参数 ---
GAMMA_PBRS = 0.99           # PBRS 折扣因子 (通常与 RL 算法的 γ 一致)

# --- 归一化常数 (用于潜力函数计算) ---
MAX_SUN_CAPACITY = 2000.0   # 阳光归一化上限
THREAT_DECAY = 0.5          # 威胁惩罚指数衰减系数 (越大 → 越惩罚靠近房子的僵尸)

# --- 植物防御价值 (用于 Φ_defense 计算) ---
# 不同植物对防线的贡献权重
PLANT_DEFENSE_VALUES = {
    "Sunflower": 0.5,       # 向日葵：经济价值，防御较低
    "Peashooter": 1.5,      # 豌豆射手：输出价值
    "Wallnut": 2.0,         # 坚果墙：高防御价值
    "Potatomine": 1.0,      # 土豆雷：爆发价值
}

# --- 僵尸威胁权重 (用于 Φ_threat 计算) ---
# 不同僵尸类型的威胁程度
ZOMBIE_THREAT_WEIGHTS = {
    "Zombie": 1.0,          # 普通僵尸
    "Zombie_cone": 1.5,     # 路障僵尸
    "Zombie_bucket": 2.5,   # 铁桶僵尸 (高威胁)
    "Zombie_flag": 1.0,     # 旗帜僵尸
}

# --- 僵尸击杀奖励权重 (用于 Φ_kills 计算) ---
# 不同僵尸类型的击杀价值 (鼓励优先消灭高威胁僵尸)
ZOMBIE_KILL_VALUES = {
    "Zombie": 1.0,          # 普通僵尸
    "Zombie_cone": 1.5,     # 路障僵尸
    "Zombie_bucket": 3.0,   # 铁桶僵尸 (高击杀价值)
    "Zombie_flag": 1.0,     # 旗帜僵尸
}