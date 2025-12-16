# ==================== PvZ 策略性奖励塑形配置 ====================
# 基于塔防游戏核心策略设计的奖励函数参数
#
# 核心策略原则:
# 1. 经济优先: 前期种向日葵建立经济，放在后排（左侧）
# 2. 响应式防御: 僵尸出现后再部署豌豆，不提前浪费
# 3. 土豆地雷: 处理紧急威胁，鼓励爆炸，惩罚浪费
# 4. 坚果墙: 仅在高压力时使用，保护前排植物
# 5. 时机把控: 鼓励土豆启动后立刻引爆

# ==================== 游戏阶段定义 ====================
# 用于区分前期/中期/后期，调整策略权重
EARLY_GAME_THRESHOLD = 80      # 前期: chrono < 80 (约40秒)
MID_GAME_THRESHOLD = 200       # 中期: 80 <= chrono < 200
# 后期: chrono >= 200

# ==================== 潜力函数权重 ====================
# 基础权重（会根据游戏阶段动态调整）
# 简化权重配置，突出核心目标
W_SUN = 0.5                    # 资源管理 (降低，避免过度追求阳光)
W_DEFENSE = 2.0                # 防御覆盖 (提高，鼓励建立防线)
W_THREAT = 3.0                 # 威胁降低 (最重要！阻止僵尸靠近)
W_KILLS = 2.5                  # 击杀奖励 (提高，鼓励消灭僵尸)
W_STRATEGY = 0.5               # 策略性奖励 (降低，减少细碎信号)

# ==================== 稀疏奖励 ====================
# 大幅增加胜负差距，让目标更明确
REWARD_WIN = 500.0             # 胜利奖励 (大幅提高)
REWARD_LOSE = -200.0           # 失败惩罚 (适度提高)
REWARD_TIMEOUT = -100.0        # 超时惩罚 (小于失败，但仍有惩罚，防止拖延)

# ==================== 密集奖励 ====================
# 存活奖励是关键: 每帧存活应该得到正奖励
# 这确保存活更久的策略总是比早死的策略好
REWARD_SURVIVAL_PER_FRAME = 1.0    # 提高存活奖励
INVALID_ACTION_PENALTY = -0.5      # 无效动作惩罚

# ==================== 生命惩罚 ====================
# 僵尸进家惩罚，让模型对防御失败有明确的负反馈
LIFE_LOSS_PENALTY = -50.0

# ==================== PBRS 参数 ====================
GAMMA_PBRS = 0.99

# ==================== Φ_sun 参数 ====================
MAX_SUN_CAPACITY = 500.0       # 降低，避免过度囤积阳光
SUNFLOWER_POTENTIAL_BONUS = 0.15   # 降低，避免过度追求向日葵

# 向日葵位置奖励: 鼓励放在后排（pos 越小越好）
# 奖励 = SUNFLOWER_POS_REWARD * (1 - pos / LANE_LENGTH)
SUNFLOWER_POSITION_REWARD = 0.15

# ==================== Φ_defense 参数 ====================
PLANT_DEFENSE_VALUES = {
    "Sunflower": 0.1,          # 向日葵几乎无防御价值
    "Peashooter": 1.5,         # 主要输出
    "Wallnut": 2.0,            # 高防御
    "Potatomine": 1.5,         # 爆发防御（已激活时）
}

# 安全缓冲距离权重
BUFFER_DISTANCE_WEIGHT = 0.2
BREACH_PENALTY_WEIGHT = 0.6

# 行覆盖奖励
LANE_COVERAGE_BONUS = 0.25

# ==================== Φ_threat 参数 ====================
THREAT_POWER = 3.0             # 提高幂次，对靠近的僵尸惩罚更重
THREAT_HP_FACTOR = True

ZOMBIE_THREAT_WEIGHTS = {
    "Zombie": 1.0,
    "Zombie_cone": 2.0,        # 提高威胁权重
    "Zombie_bucket": 4.0,      # 铁桶威胁最大
    "Zombie_flag": 1.5,
}

# ==================== Φ_kills 参数 ====================
ZOMBIE_KILL_VALUES = {
    "Zombie": 1.0,
    "Zombie_cone": 2.5,
    "Zombie_bucket": 5.0,      # 击杀铁桶奖励最高
    "Zombie_flag": 2.0,
}
MAX_KILLS_FOR_NORMALIZATION = 30.0

# 即时击杀奖励倍数: kill_value * KILL_REWARD_MULTIPLIER
# 让有击杀的策略明显优于无击杀的
KILL_REWARD_MULTIPLIER = 5.0   # 提高击杀奖励

# ==================== 策略性奖励参数 ====================

# --- 向日葵策略 ---
# 前期种向日葵奖励（鼓励早期经济建设）
SUNFLOWER_EARLY_GAME_BONUS = 0.5
# 向日葵理想位置: pos < SUNFLOWER_IDEAL_POS 时给额外奖励
SUNFLOWER_IDEAL_POS = 3
SUNFLOWER_GOOD_POS_BONUS = 0.3
# 向日葵放在危险位置(有僵尸的行且pos较大)的惩罚
SUNFLOWER_DANGER_PENALTY = -0.4

# --- 豌豆射手策略 ---
# 在有僵尸的行放豌豆奖励（响应式防御）
PEASHOOTER_REACTIVE_BONUS = 0.4
# 在没有僵尸的行提前放豌豆的惩罚（前期浪费资源）
# 注意: 惩罚不要太重，否则种植物反而不如不种
PEASHOOTER_PREMATURE_PENALTY = -0.05
# 豌豆射手理想位置: 中间偏后，既能输出又有缓冲
PEASHOOTER_IDEAL_POS_MIN = 2
PEASHOOTER_IDEAL_POS_MAX = 5

# --- 土豆地雷策略 ---
# 土豆地雷激活时间 (秒)
POTATOMINE_ACTIVATION_TIME = 14  # 与游戏设置一致
# 土豆爆炸奖励（成功消灭僵尸）
POTATOMINE_EXPLODE_BONUS = 2.0
# 土豆爆炸消灭多个僵尸的额外奖励
POTATOMINE_MULTI_KILL_BONUS = 1.5
# 土豆未爆炸就死亡的惩罚（被其他植物保护或位置不当）
POTATOMINE_WASTED_PENALTY = -1.5
# 土豆放置在高威胁位置的奖励（鼓励处理紧急情况）
POTATOMINE_EMERGENCY_BONUS = 0.8
# 土豆启动后快速引爆的奖励（激活后几帧内爆炸）
POTATOMINE_QUICK_EXPLODE_BONUS = 0.5
POTATOMINE_QUICK_EXPLODE_FRAMES = 20  # 激活后20帧内爆炸算"快速"

# --- 坚果墙策略 ---
# 坚果墙仅在高压力时使用的奖励
WALLNUT_HIGH_PRESSURE_BONUS = 0.6
# 定义"高压力": 该行有僵尸且距离 < WALLNUT_PRESSURE_THRESHOLD
WALLNUT_PRESSURE_THRESHOLD = 4
# 坚果墙放在前排的奖励 (pos >= WALLNUT_FRONTLINE_POS)
WALLNUT_FRONTLINE_POS = 4
WALLNUT_FRONTLINE_BONUS = 0.4
# 坚果墙保护了该行其他植物的奖励
WALLNUT_PROTECT_BONUS = 0.3
# 低压力时放坚果的惩罚（浪费资源）
# 注意: 惩罚不要太重，否则种植物反而不如不种  
WALLNUT_LOW_PRESSURE_PENALTY = -0.1

# ==================== 动作即时奖励 ====================
# 这些奖励在执行动作时立即给予，而非通过潜力函数

# 成功放置植物的基础奖励（鼓励种植物比什么都不做好）
PLANT_PLACEMENT_REWARD = 0.5

# 在紧急情况下正确响应的奖励
EMERGENCY_RESPONSE_BONUS = 0.5

# ==================== 纠正参数 ====================
CORRECT_ESCAPE_SCORE = True

# ==================== 旧版V2环境兼容参数 ====================
# 这些参数用于旧环境，保留以确保兼容性

# 威胁衰减系数（V2环境使用）
THREAT_DECAY = 0.5

# 防线参数（V2环境使用）
DEFENSE_PREP_BONUS_PER_PLANT = 0.02
DEFENSE_FRONTLINE_WEIGHT = 0.6
DEFENSE_VALUE_WEIGHT = 0.4
CORRECT_ESCAPE_SCORE = True
