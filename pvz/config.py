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


# ==================== 奖励函数参数 ====================
# 所有奖励相关参数已移至 reward_config.py
# 请使用: from pvz import reward_config
# 或在环境中: from pvz import reward_config as rcfg
#
# 为了向后兼容，延迟导入（避免循环导入和启动时阻塞）
def _load_reward_config():
    try:
        from . import reward_config as _rcfg
        globals().update({k: v for k, v in _rcfg.__dict__.items() if not k.startswith('_')})
    except (ImportError, Exception):
        pass

_load_reward_config()

