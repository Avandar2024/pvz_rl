"""
PVZ Environment V3

基于 V2 版本，添加了：
1. 人工设计的局面评估函数作为奖励塑形 (reward shaping)
2. 种植位置引导奖励，鼓励合理布局
3. 更精细的奖励结构
"""

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
from pvz import Scene, WaveZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut, Potatomine
from pvz import reward_config as rcfg
import numpy as np

from .position_evaluator import (
    evaluate_lane,
    SHAPING_COEFFICIENT
)

# 种植位置奖励配置（直接从 reward_config 导入）
PLACEMENT_REWARDS = rcfg.PLACEMENT_REWARDS
PLACEMENT_DEFAULT = rcfg.PLACEMENT_DEFAULT
SUN_WASTE_PENALTY = rcfg.SUN_WASTE_PENALTY
FRONT_ROW_PENALTY = rcfg.FRONT_ROW_PENALTY


def _get_placement_reward(plant_type: str, col: int) -> float:
    """获取种植位置奖励（在 env 中直接计算）"""
    plant_rewards = PLACEMENT_REWARDS.get(plant_type, {})
    default_reward = PLACEMENT_DEFAULT.get(plant_type, 0)
    return plant_rewards.get(col, default_reward)

MAX_ZOMBIE_HP = 10000
MAX_SUN = 10000
MAX_COOLDOWN = 20  # Potatomine/Wallnut


class PVZEnv_V3(gym.Env):
    """
    PvZ强化学习环境 V3
    
    相比V2的改进:
    - 添加局面评估差分奖励 (势能塑形)
    - 添加种植位置引导奖励
    - 更好的奖励结构设计
    """
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.plant_deck = {
            "sunflower": Sunflower, 
            "peashooter": Peashooter, 
            "wall-nut": Wallnut,
            "potatomine": Potatomine
        }

        self.action_space = Discrete(len(self.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)
        
        grid_size = config.N_LANES * config.LANE_LENGTH
        nvec = ([len(self.plant_deck) + 1] * grid_size + 
                [MAX_ZOMBIE_HP] * grid_size + 
                [MAX_SUN] + 
                [2] * len(self.plant_deck))
        self.observation_space = MultiDiscrete(nvec)

        self._plant_names = [plant_name for plant_name in self.plant_deck]
        self._plant_classes = [self.plant_deck[plant_name].__name__ for plant_name in self.plant_deck]
        self._plant_no = {self._plant_classes[i]: i for i in range(len(self._plant_names))}
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._reward = 0

    def step(self, action):
        """
        执行一步动作
        
        奖励组成:
        1. base_reward: 基础环境奖励 (击杀分数等)
        2. placement_reward: 种植位置奖励 (引导合理布局)
        3. shaping_reward: 局面评估差分奖励 (势能塑形，只评估种植行)
        4. terminal_reward: 终局奖励 (胜利/失败/超时)
        
        返回:
            obs, reward, terminated, truncated, info
        """
        placement_reward = 0.0
        shaping_reward = 0.0
        old_lane_value = 0.0
        plant_lane = -1
        planted = False
        
        if action > 0:
            # 解析种植动作: action = 1 + cell_idx * n_plants + plant_idx
            # cell_idx = lane + pos * N_LANES
            action_parsed = action - 1
            n_plants = len(self.plant_deck)
            plant_idx = action_parsed % n_plants
            cell_idx = action_parsed // n_plants
            lane = cell_idx % config.N_LANES
            pos = cell_idx // config.N_LANES
            plant_name = self._plant_names[plant_idx]
            
            # 动作由 mask_available_actions 保证有效，直接执行
            # 动作前评估该行
            old_lane_value = evaluate_lane(self._scene, lane)
            plant_lane = lane
            
            # 执行种植
            move = Move(plant_name, lane, pos)
            move.apply_move(self._scene)
            planted = True
            
            # 计算位置奖励
            placement_reward = _get_placement_reward(plant_name, pos)
            
            # 检查该行是否有僵尸
            has_zombie_in_lane = any(z.lane == lane and z.hp > 0 for z in self._scene.zombies)
            
            # 阳光浪费惩罚：非向日葵在没有僵尸的行种植
            if plant_name != 'sunflower' and not has_zombie_in_lane:
                placement_reward += SUN_WASTE_PENALTY
            
            # 第一排（pos=0）种植非向日葵的额外惩罚
            if pos == 0 and plant_name != 'sunflower':
                placement_reward += FRONT_ROW_PENALTY
        
        # 场景步进
        self._scene.step()
        base_reward = self._scene.score
        
        # 检查结束条件
        is_victory = self._scene.is_victory()
        is_defeat = self._scene.is_defeat()
        terminated = is_defeat or is_victory
        truncated = self._scene.is_timeout()
        
        # 继续步进直到可以执行下一个动作或游戏结束
        while (not self._scene.move_available()) and (not terminated) and (not truncated):
            self._scene.step()
            base_reward += self._scene.score
            is_victory = self._scene.is_victory()
            is_defeat = self._scene.is_defeat()
            terminated = is_defeat or is_victory
            truncated = self._scene.is_timeout()
        
        # 计算势能塑形奖励（只有种植成功才计算）
        if planted and plant_lane >= 0:
            new_lane_value = evaluate_lane(self._scene, plant_lane)
            shaping_reward = SHAPING_COEFFICIENT * (new_lane_value - old_lane_value)
        
        # 终局奖励
        terminal_reward = 0.0
        if is_victory:
            terminal_reward = rcfg.REWARD_WIN
        elif is_defeat:
            terminal_reward = rcfg.REWARD_LOSE
        elif truncated:
            terminal_reward = rcfg.REWARD_TIMEOUT
        
        # 总奖励
        total_reward = (
            base_reward * 1.0 +
            placement_reward * 1.0 +
            shaping_reward * 1.0 +
            terminal_reward * 1.0
        )
        
        # 观察
        obs = self._get_obs()
        self._reward = total_reward
        
        # 额外信息
        info = {
            'base_reward': base_reward,
            'placement_reward': placement_reward,
            'shaping_reward': shaping_reward,
            'terminal_reward': terminal_reward,
            'plant_lane': plant_lane
        }
        
        return obs, total_reward, terminated, truncated, info

    def _get_obs(self):
        """获取观察向量"""
        obs_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        zombie_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        
        for plant in self._scene.plants:
            obs_grid[plant.lane * config.LANE_LENGTH + plant.pos] = (
                self._plant_no[plant.__class__.__name__] + 1
            )
        
        for zombie in self._scene.zombies:
            zombie_grid[zombie.lane * config.LANE_LENGTH + zombie.pos] += zombie.hp
        
        action_available = np.array([
            self._scene.plant_cooldowns[plant_name] <= 0 
            for plant_name in self.plant_deck
        ])
        action_available *= np.array([
            self._scene.sun >= self.plant_deck[plant_name].COST 
            for plant_name in self.plant_deck
        ])
        
        return np.concatenate([
            obs_grid, 
            zombie_grid, 
            [min(self._scene.sun, MAX_SUN)], 
            action_available
        ])

    def reset(self, seed=None, options=None):
        """重置环境"""
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        obs = self._get_obs()
        return obs, {}

    def render(self, mode='human'):
        """渲染环境"""
        print(self._scene)
        print(f"Score since last action: {self._reward:.2f}")

    def close(self):
        pass

    def _take_action(self, action):
        """执行动作（种植逻辑已在 step 中处理，此方法保留为空操作处理）"""
        # 种植动作已在 step() 中直接处理
        # 此方法保留用于 action == 0 (no-op) 的情况
        pass

    def mask_available_actions(self):
        """获取可用动作掩码"""
        empty_cells, available_plants = self._scene.get_available_moves()
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * len(self.plant_deck)
        for plant in available_plants:
            idx = empty_cells + self._plant_no[plant.__name__] + 1
            mask[idx] = True
        return mask

    def num_observations(self):
        """观察空间维度"""
        return 2 * config.N_LANES * config.LANE_LENGTH + len(self.plant_deck) + 1