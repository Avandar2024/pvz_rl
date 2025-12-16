"""
局面评估模块

用于评估当前游戏状态的价值，作为奖励塑形(reward shaping)使用。
评估基于人工设计的规则，考虑植物布局、僵尸威胁、经济状况等因素。

使用 Numba JIT 加速模拟计算。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numba import njit

from pvz import config, reward_config
from pvz.entities.plants.sunflower import Sunflower
from pvz.entities.plants.peashooter import Peashooter, PEASHOOTER_ATTACK
from pvz.entities.plants.wallnut import Wallnut
from pvz.entities.plants.potatomine import Potatomine

# 从 reward_config 导入奖励参数
LANE_SCORE_MAX = reward_config.LANE_SCORE_MAX
SHAPING_COEFFICIENT = reward_config.SHAPING_COEFFICIENT

# ==================== Numba 加速常量 ====================
# 植物类型编码
PLANT_TYPE_SUNFLOWER = 0
PLANT_TYPE_PEASHOOTER = 1
PLANT_TYPE_WALLNUT = 2
PLANT_TYPE_POTATOMINE = 3
PLANT_TYPE_UNKNOWN = -1

# 游戏参数 (编译时常量)
_FPS = config.FPS
_LANE_LENGTH = config.LANE_LENGTH
_PEA_SPEED_PER_FRAME = 5.0 / _FPS  # 豌豆每帧移动格数
_PEA_ATTACK = float(PEASHOOTER_ATTACK)
_PEASHOOTER_COOLDOWN = int(1.5 * _FPS) - 1  # 豌豆射手攻击冷却

@dataclass
class SimulatedZombie:
    """模拟用的僵尸状态
    
    原游戏逻辑:
    - _cell_length = WALKING_SPEED * FPS = 5 * 2 = 10 帧穿过一格
    - _offset 初始为 _cell_length - 1 = 9，每帧减1，到0时pos-=1
    
    实现设计:
    - 保存原始offset值 (整数 0 到 cell_length-1)
    - 需要比较时再归一化为 offset/cell_length (避免浮点误差)
    """
    hp: float
    pos: int
    offset: int  # 原始offset值 (0 到 cell_length-1)
    cell_length: int  # 穿过一个格子需要的帧数 (_cell_length)
    attack_per_frame: float  # 每帧攻击伤害
    
    @property
    def exact_pos(self) -> float:
        """精确位置用于比较 (pos, offset) 的大小"""
        return self.pos + self.offset / self.cell_length


@dataclass
class SimulatedPlant:
    """模拟用的植物状态"""
    hp: float
    pos: int
    plant_type: str
    attack_cooldown: int = 0  # 攻击/激活冷却（帧数）


@dataclass
class SimulatedProjectile:
    """模拟用的子弹状态
    
    原游戏逻辑:
    - _pos: 整数格子位置
    - _offset: 0-1之间的小数，表示在格子内的偏移
    - 每帧移动: offset += speed/FPS, pos += int(offset), offset -= int(offset)
    - 命中检测: 检查(previous_pos, previous_offset) 到 (pos, offset) 路径上是否有僵尸
    """
    pos: int  # 整数格子位置
    offset: float  # 0-1之间的偏移
    prev_pos: int  # 上一帧位置
    prev_offset: float  # 上一帧偏移
    attack: float
    speed_per_frame: float  # speed / FPS


@dataclass
class SimulationResult:
    """单行模拟结果"""
    zombies_defeated: bool      # 僵尸是否被全部消灭
    plants_eaten: int           # 被吃掉的植物数量
    breakthrough: bool          # 是否会突破防线（僵尸到达pos<0）
    breakthrough_time: int      # 突破需要的帧数（越大越好，表示拖延时间长）

    simulation_frames: int      # 模拟总帧数


def get_plant_type_name(plant) -> str:
    """获取植物类型名称"""
    if isinstance(plant, Sunflower):
        return 'sunflower'
    elif isinstance(plant, Peashooter):
        return 'peashooter'
    elif isinstance(plant, Wallnut):
        return 'wallnut'
    elif isinstance(plant, Potatomine):
        return 'potatomine'
    return 'unknown'


# Numba 加速的模拟函数

@njit
def _simulate_lane_numba(
    zombie_hp: np.ndarray,           # float64[n_zombies]
    zombie_pos: np.ndarray,          # int32[n_zombies]
    zombie_offset: np.ndarray,       # int32[n_zombies]
    zombie_cell_length: np.ndarray,  # int32[n_zombies]
    zombie_attack: np.ndarray,       # float64[n_zombies]
    plant_hp: np.ndarray,            # float64[n_plants]
    plant_pos: np.ndarray,           # int32[n_plants]
    plant_type: np.ndarray,          # int32[n_plants] (0=sunflower, 1=peashooter, 2=wallnut, 3=potato)
    plant_cooldown: np.ndarray,      # int32[n_plants]
) -> Tuple[bool, int, bool, int, float, int]:
    """
    Numba 加速的单行模拟
    
    返回: (zombies_defeated, plants_eaten, breakthrough, breakthrough_time, frames)
    """
    n_zombies = len(zombie_hp)
    n_plants = len(plant_hp)
    initial_plant_count = n_plants
    
    # 子弹状态 (预分配：每个豌豆射手最多同时2颗子弹，一行最多9个豌豆射手)
    max_projectiles = 100
    proj_pos = np.zeros(max_projectiles, dtype=np.int32)
    proj_offset = np.zeros(max_projectiles, dtype=np.float64)
    proj_prev_pos = np.zeros(max_projectiles, dtype=np.int32)
    proj_prev_offset = np.zeros(max_projectiles, dtype=np.float64)
    proj_active = np.zeros(max_projectiles, dtype=np.bool_)
    n_projectiles = 0
    
    frame = 0
    
    while True:
        frame += 1
        
        # 1. 豌豆射手发射子弹
        for i in range(n_plants):
            if plant_hp[i] <= 0:
                continue
            if plant_type[i] == PLANT_TYPE_PEASHOOTER:
                if plant_cooldown[i] <= 0:
                    # 发射子弹
                    if n_projectiles < max_projectiles:
                        proj_pos[n_projectiles] = plant_pos[i]
                        proj_offset[n_projectiles] = 0.0
                        proj_prev_pos[n_projectiles] = plant_pos[i]
                        proj_prev_offset[n_projectiles] = 0.0
                        proj_active[n_projectiles] = True
                        n_projectiles += 1
                    plant_cooldown[i] = _PEASHOOTER_COOLDOWN
                else:
                    plant_cooldown[i] -= 1
        
        # 2. 子弹移动和命中检测
        for p in range(n_projectiles):
            if not proj_active[p]:
                continue
            
            # 保存前一帧位置
            proj_prev_pos[p] = proj_pos[p]
            proj_prev_offset[p] = proj_offset[p]
            
            # 移动
            proj_offset[p] += _PEA_SPEED_PER_FRAME
            move_cells = int(proj_offset[p])
            proj_pos[p] += move_cells
            proj_offset[p] -= move_cells
            
            # 检查出界
            if proj_pos[p] >= _LANE_LENGTH:
                proj_active[p] = False
                continue
            
            # 命中检测
            prev_exact = proj_prev_pos[p] + proj_prev_offset[p]
            curr_exact = proj_pos[p] + proj_offset[p]
            
            hit_idx = -1
            hit_min_pos = 1e9
            
            for z in range(n_zombies):
                if zombie_hp[z] <= 0:
                    continue
                z_exact = zombie_pos[z] + zombie_offset[z] / zombie_cell_length[z]
                
                if prev_exact <= z_exact <= curr_exact:
                    if z_exact < hit_min_pos:
                        hit_min_pos = z_exact
                        hit_idx = z
            
            if hit_idx >= 0:
                zombie_hp[hit_idx] -= _PEA_ATTACK
                proj_active[p] = False
        
        # 3. 土豆地雷检测
        for i in range(n_plants):
            if plant_hp[i] <= 0:
                continue
            if plant_type[i] == PLANT_TYPE_POTATOMINE:
                if plant_cooldown[i] <= 0:
                    # 检查是否有僵尸踩到
                    triggered = False
                    for z in range(n_zombies):
                        if zombie_hp[z] > 0 and zombie_pos[z] == plant_pos[i]:
                            triggered = True
                            break
                    
                    if triggered:
                        # 爆炸：消灭同位置所有僵尸
                        for z in range(n_zombies):
                            if zombie_pos[z] == plant_pos[i]:
                                zombie_hp[z] = 0
                        plant_hp[i] = 0
                else:
                    plant_cooldown[i] -= 1
        
        # 4. 僵尸移动/攻击
        for z in range(n_zombies):
            if zombie_hp[z] <= 0:
                continue
            
            # 检查是否有植物阻挡
            blocking_plant = -1
            for i in range(n_plants):
                if plant_hp[i] > 0 and plant_pos[i] == zombie_pos[z]:
                    blocking_plant = i
                    break
            
            if blocking_plant >= 0:
                # 啃咬植物
                plant_hp[blocking_plant] -= zombie_attack[z]
            else:
                # 向左移动
                if zombie_offset[z] <= 0:
                    zombie_pos[z] -= 1
                    zombie_offset[z] = zombie_cell_length[z] - 1
                else:
                    zombie_offset[z] -= 1
        
        # 5. 检查终止条件
        
        # 5a. 统计存活僵尸
        alive_zombies = 0
        for i in range(n_zombies):
            if zombie_hp[i] > 0:
                alive_zombies += 1
        
        if alive_zombies == 0:
            # 僵尸全灭
            alive_plants = 0
            for i in range(n_plants):
                if plant_hp[i] > 0:
                    alive_plants += 1
            plants_eaten = initial_plant_count - alive_plants
            return (True, plants_eaten, False, 0, frame)
        
        # 5b. 检查是否突破
        leftmost_pos = _LANE_LENGTH + 1
        leftmost_exact = float(_LANE_LENGTH + 1)
        for z in range(n_zombies):
            if zombie_hp[z] > 0:
                z_exact = zombie_pos[z] + zombie_offset[z] / zombie_cell_length[z]
                if z_exact < leftmost_exact:
                    leftmost_exact = z_exact
                    leftmost_pos = zombie_pos[z]
        
        if leftmost_pos < 0:
            alive_plants = 0
            for i in range(n_plants):
                if plant_hp[i] > 0:
                    alive_plants += 1
            plants_eaten = initial_plant_count - alive_plants
            return (False, plants_eaten, True, frame, frame)


def simulate_lane(plants_in_lane: List, zombies_in_lane: List) -> SimulationResult:
    """
    模拟单行的僵尸vs植物战斗（使用 Numba 加速）。
    
    参数:
        plants_in_lane: 该行的植物列表 (从scene.plants筛选)
        zombies_in_lane: 该行的僵尸列表 (从scene.zombies筛选)
    
    返回:
        SimulationResult 包含模拟结果
    """
    if not zombies_in_lane:
        # 无僵尸，直接返回安全状态
        return SimulationResult(
            zombies_defeated=True,
            plants_eaten=0,
            breakthrough=False,
            breakthrough_time=0,
            simulation_frames=0
        )
    
    n_zombies = len(zombies_in_lane)
    n_plants = len(plants_in_lane)
    
    # 转换僵尸数据为 numpy 数组
    zombie_hp = np.zeros(n_zombies, dtype=np.float64)
    zombie_pos = np.zeros(n_zombies, dtype=np.int32)
    zombie_offset = np.zeros(n_zombies, dtype=np.int32)
    zombie_cell_length = np.zeros(n_zombies, dtype=np.int32)
    zombie_attack = np.zeros(n_zombies, dtype=np.float64)
    
    for i, z in enumerate(zombies_in_lane):
        zombie_hp[i] = float(z.hp)
        zombie_pos[i] = z.pos
        zombie_offset[i] = z._offset
        zombie_cell_length[i] = z._cell_length
        zombie_attack[i] = z._attack
    
    # 转换植物数据为 numpy 数组
    if n_plants > 0:
        plant_hp = np.zeros(n_plants, dtype=np.float64)
        plant_pos = np.zeros(n_plants, dtype=np.int32)
        plant_type = np.zeros(n_plants, dtype=np.int32)
        plant_cooldown = np.zeros(n_plants, dtype=np.int32)
        
        for i, p in enumerate(plants_in_lane):
            plant_hp[i] = float(p.hp)
            plant_pos[i] = p.pos
            
            # 植物类型编码
            if isinstance(p, Sunflower):
                plant_type[i] = PLANT_TYPE_SUNFLOWER
            elif isinstance(p, Peashooter):
                plant_type[i] = PLANT_TYPE_PEASHOOTER
                plant_cooldown[i] = int(p.attack_cooldown)
            elif isinstance(p, Wallnut):
                plant_type[i] = PLANT_TYPE_WALLNUT
            elif isinstance(p, Potatomine):
                plant_type[i] = PLANT_TYPE_POTATOMINE
                plant_cooldown[i] = int(p.attack_cooldown)
            else:
                plant_type[i] = PLANT_TYPE_UNKNOWN
    else:
        # 空数组
        plant_hp = np.zeros(0, dtype=np.float64)
        plant_pos = np.zeros(0, dtype=np.int32)
        plant_type = np.zeros(0, dtype=np.int32)
        plant_cooldown = np.zeros(0, dtype=np.int32)
    
    # 调用 Numba 加速函数
    result = _simulate_lane_numba(
        zombie_hp, zombie_pos, zombie_offset, zombie_cell_length, zombie_attack,
        plant_hp, plant_pos, plant_type, plant_cooldown
    )
    
    zombies_defeated, plants_eaten, breakthrough, breakthrough_time, frames = result
    
    return SimulationResult(
        zombies_defeated=zombies_defeated,
        plants_eaten=plants_eaten,
        breakthrough=breakthrough,
        breakthrough_time=breakthrough_time,
        simulation_frames=frames
    )


# 局面评估函数

def evaluate_lane_peaceful(plants_in_lane: List) -> float:
    """
    无僵尸时评估植物配置
    """
    score = 0.0
    
    # 位置合理性加分
    for p in plants_in_lane:
        plant_type = get_plant_type_name(p)
        col = p.pos
        
        if plant_type == 'sunflower' and col <= 1:
            score += 10  # 向日葵在前两列
        if plant_type == 'peashooter' and 1 <= col <= 3:
            score += 6   # 豌豆在2-4列
        if plant_type == 'wallnut' and col >= 4:
            score += 6   # 坚果在后排
    
    return score


def evaluate_lane_with_zombies(plants_in_lane: List, zombies_in_lane: List) -> float:
    """
    有僵尸时评估单行局面
    
    返回: 0-150分
    """
    sim_result = simulate_lane(plants_in_lane, zombies_in_lane)
    
    if sim_result.zombies_defeated and sim_result.plants_eaten == 0:
        # 情况A: 僵尸被完全消灭，植物无损失 (最好)
        score = 130.0
        return min(score, 150.0)
    
    elif not sim_result.breakthrough:
        # 情况B: 僵尸会吃掉一些植物，但不会突破（模拟中消灭）
        score = 120.0 - sim_result.plants_eaten * 15.0
        return max(score, 40.0)
    
    else:
        # 情况C: 僵尸会突破防线 (最差)
        # 但拖延时间长、僵尸停留位置靠右会稍微好一点
        score = 0.0
        
        # 时间加分: 每拖延10帧 +0.5分，最多+15分
        time_bonus = min(sim_result.breakthrough_time / 20.0, 15.0)
        score += time_bonus
        
        # 确保突破情况分数始终低于不突破
        return min(score, 35.0)


def evaluate_lane(scene, lane: int) -> float:
    """
    评估单行得分
    
    参数:
        scene: 游戏场景对象
        lane: 行号 (0-4)
    
    返回: 0-150分
    """
    # 筛选该行的植物和僵尸
    plants_in_lane = [p for p in scene.plants if p.lane == lane and p.hp > 0]
    zombies_in_lane = [z for z in scene.zombies if z.lane == lane and z.hp > 0]
    
    if not zombies_in_lane:
        # 无僵尸，评估植物布局
        return evaluate_lane_peaceful(plants_in_lane)
    else:
        # 有僵尸，模拟战斗
        return evaluate_lane_with_zombies(plants_in_lane, zombies_in_lane)
