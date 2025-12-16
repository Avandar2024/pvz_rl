# ==================== 游戏基础参数 ====================
FPS = 2                     # 游戏帧率
MAX_FRAMES = 1000            # 最大游戏帧数

N_LANES = 5                 # 行数 (高度)
LANE_LENGTH = 9             # 列数 (宽度)

INITIAL_SUN_AMOUNT = 50     # 初始阳光数量

# 天然阳光产出
NATURAL_SUN_PRODUCTION = 25           # 每次产出阳光数量
NATURAL_SUN_PRODUCTION_COOLDOWN = 10  # 产出间隔 (秒)

# 割草机
MOWERS = False              # 是否启用割草机

# ==================== 奖励函数参数 ====================
# 所有奖励相关参数已移至 reward_config.py
# 请使用: from pvz import reward_config
# 或在环境中: from pvz import reward_config as rcfg
#
# 为了向后兼容，延迟导入（避免循环导入）
def _load_reward_config():
    try:
        from . import reward_config as _rcfg
        # 导出常用的奖励参数到 config 命名空间（向后兼容）
        globals()['SURVIVAL'] = _rcfg.SURVIVAL_REWARD
        globals()['SURVIVAL_STEP'] = _rcfg.SURVIVAL_STEP
        globals()['SCORE_ALIVE_PLANT'] = _rcfg.PLANT_ALIVE_REWARD
        globals()['SCORE_ALIVE_MOWER'] = _rcfg.MOWER_ALIVE_REWARD
    except (ImportError, Exception):
        # 默认值（如果 reward_config 不存在）
        globals()['SURVIVAL'] = 0
        globals()['SURVIVAL_STEP'] = 20
        globals()['SCORE_ALIVE_PLANT'] = 0
        globals()['SCORE_ALIVE_MOWER'] = 0

_load_reward_config()