"""
策略性奖励塑形环境 V3
基于 PvZ 核心策略设计:
1. 经济: 前期多种向日葵，放后排
2. 防御: 响应式部署豌豆，僵尸来了再种
3. 土豆: 处理紧急威胁，鼓励爆炸，惩罚浪费
4. 坚果: 高压力时保护前排
"""

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
from pvz import Scene, WaveZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut, Potatomine
import numpy as np

try:
    from pvz import reward_config as rcfg
except ImportError:
    from pvz import config as rcfg

MAX_ZOMBIE_HP = 10000
MAX_SUN = 10000


class PVZEnv_V4(gym.Env):
    """策略性奖励塑形的 PvZ 环境"""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.plant_deck = {
            "sunflower": Sunflower,
            "peashooter": Peashooter,
            "wall-nut": Wallnut,
            "potatomine": Potatomine
        }

        # PBRS 状态
        self._prev_potential = 0.0
        self._total_kills = 0
        self._weighted_kills = 0.0
        
        # 土豆地雷追踪
        self._potatomine_tracker = {}  # id -> {placed_frame, activated_frame, lane, pos}
        
        # 策略奖励累积
        self._strategy_rewards = 0.0

        self.action_space = Discrete(len(self.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)

        grid_size = config.N_LANES * config.LANE_LENGTH
        nvec = ([len(self.plant_deck) + 1] * grid_size +
                [MAX_ZOMBIE_HP] * grid_size +
                [MAX_SUN] +
                [2] * len(self.plant_deck))
        self.observation_space = MultiDiscrete(nvec)

        self._plant_names = list(self.plant_deck.keys())
        self._plant_classes = [self.plant_deck[name].__name__ for name in self.plant_deck]
        self._plant_no = {cls: i for i, cls in enumerate(self._plant_classes)}
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._reward = 0

    # ==================== 游戏阶段判断 ====================
    
    def _get_game_phase(self):
        """返回当前游戏阶段: 'early', 'mid', 'late'"""
        chrono = self._scene._chrono
        early = getattr(rcfg, 'EARLY_GAME_THRESHOLD', 80)
        mid = getattr(rcfg, 'MID_GAME_THRESHOLD', 200)
        
        if chrono < early:
            return 'early'
        elif chrono < mid:
            return 'mid'
        return 'late'
    
    def _get_lane_pressure(self, lane):
        """计算某行的压力程度 (0=无压力, 1=高压力)"""
        zombies_in_lane = [z for z in self._scene.zombies if z.lane == lane]
        if not zombies_in_lane:
            return 0.0
        
        # 找最靠近房子的僵尸
        min_pos = min(z.pos + z.get_offset() for z in zombies_in_lane)
        # 压力与距离成反比
        pressure = 1.0 - (min_pos / config.LANE_LENGTH)
        return max(0.0, min(1.0, pressure))

    # ==================== 潜力函数 ====================
    
    def _compute_potential(self):
        """计算总潜力 Φ(s)"""
        phase = self._get_game_phase()
        
        phi_sun = self._potential_sun()
        phi_defense = self._potential_defense()
        phi_threat = self._potential_threat()
        phi_kills = self._potential_kills()
        phi_strategy = self._potential_strategy()
        
        # 根据阶段调整权重
        w_sun = getattr(rcfg, 'W_SUN', 2.0)
        w_defense = getattr(rcfg, 'W_DEFENSE', 3.0)
        w_threat = getattr(rcfg, 'W_THREAT', 5.0)
        w_kills = getattr(rcfg, 'W_KILLS', 1.5)
        w_strategy = getattr(rcfg, 'W_STRATEGY', 3.0)
        
        if phase == 'early':
            # 前期: 经济和策略更重要
            w_sun *= 1.5
            w_strategy *= 1.3
            w_defense *= 0.8
        elif phase == 'late':
            # 后期: 威胁和防御更重要
            w_sun *= 0.6
            w_threat *= 1.2
            w_defense *= 1.2
        
        return (w_sun * phi_sun +
                w_defense * phi_defense +
                w_threat * phi_threat +
                w_kills * phi_kills +
                w_strategy * phi_strategy)

    def _potential_sun(self):
        """资源管理潜力: 阳光 + 向日葵经济价值"""
        max_sun = getattr(rcfg, 'MAX_SUN_CAPACITY', 800.0)
        sf_bonus = getattr(rcfg, 'SUNFLOWER_POTENTIAL_BONUS', 0.2)
        pos_reward = getattr(rcfg, 'SUNFLOWER_POSITION_REWARD', 0.15)
        
        sun_norm = min(self._scene.sun, max_sun) / max_sun
        
        sunflower_value = 0.0
        for p in self._scene.plants:
            if p.__class__.__name__ == "Sunflower":
                # 基础价值
                sunflower_value += sf_bonus
                # 位置奖励: pos越小(越靠近房子)越安全
                pos_factor = 1.0 - (p.pos / config.LANE_LENGTH)
                sunflower_value += pos_reward * pos_factor
        
        return sun_norm + sunflower_value

    def _potential_defense(self):
        """防御覆盖潜力: 安全缓冲距离 + 行覆盖"""
        plant_values = getattr(rcfg, 'PLANT_DEFENSE_VALUES', {})
        buffer_weight = getattr(rcfg, 'BUFFER_DISTANCE_WEIGHT', 0.2)
        breach_weight = getattr(rcfg, 'BREACH_PENALTY_WEIGHT', 0.6)
        lane_bonus = getattr(rcfg, 'LANE_COVERAGE_BONUS', 0.25)
        
        plants_by_lane = {lane: [] for lane in range(config.N_LANES)}
        zombies_by_lane = {lane: [] for lane in range(config.N_LANES)}
        
        for p in self._scene.plants:
            plants_by_lane[p.lane].append(p)
        for z in self._scene.zombies:
            zombies_by_lane[z.lane].append(z)
        
        defense_score = 0.0
        defended_lanes = 0
        
        for lane in range(config.N_LANES):
            lane_plants = plants_by_lane[lane]
            lane_zombies = zombies_by_lane[lane]
            
            # 只统计防御型植物
            defense_plants = [p for p in lane_plants
                            if p.__class__.__name__ in ("Peashooter", "Wallnut", "Potatomine")]
            
            if not defense_plants:
                if lane_zombies:
                    min_pos = min(z.pos + z.get_offset() for z in lane_zombies)
                    defense_score -= breach_weight * (1.0 - min_pos / config.LANE_LENGTH)
                continue
            
            defended_lanes += 1
            
            # 防线强度
            for p in defense_plants:
                ptype = p.__class__.__name__
                # 土豆只有激活后才有防御价值
                if ptype == "Potatomine":
                    if hasattr(p, 'attack_cooldown') and p.attack_cooldown <= 0:
                        defense_score += plant_values.get(ptype, 1.0) * 0.1
                else:
                    defense_score += plant_values.get(ptype, 1.0) * 0.1
            
            # 安全缓冲
            front_plant_pos = max(p.pos for p in defense_plants)
            
            if lane_zombies:
                front_zombie_pos = min(z.pos + z.get_offset() for z in lane_zombies)
                buffer = front_zombie_pos - front_plant_pos
                
                if buffer >= 0:
                    defense_score += buffer_weight * (buffer / config.LANE_LENGTH)
                else:
                    defense_score += breach_weight * (buffer / config.LANE_LENGTH)
        
        defense_score += defended_lanes * lane_bonus / config.N_LANES
        # 不要 clip 到 [-1, 1]，让防御收益更明显
        return defense_score

    def _potential_threat(self):
        """威胁潜力: 僵尸越近威胁越大"""
        if not self._scene.zombies:
            return 0.0
        
        threat_weights = getattr(rcfg, 'ZOMBIE_THREAT_WEIGHTS', {})
        power = getattr(rcfg, 'THREAT_POWER', 2.5)
        use_hp = getattr(rcfg, 'THREAT_HP_FACTOR', True)
        
        total_threat = 0.0
        
        for zombie in self._scene.zombies:
            actual_pos = zombie.pos + zombie.get_offset()
            pos_danger = 1.0 - (actual_pos / config.LANE_LENGTH)
            pos_danger = max(0.0, min(1.0, pos_danger))
            
            danger = pos_danger ** power
            
            ztype = zombie.__class__.__name__
            weight = threat_weights.get(ztype, 1.0)
            
            hp_factor = (zombie.hp / zombie.MAX_HP) if use_hp else 1.0
            
            total_threat += weight * hp_factor * danger
        
        max_threat = config.N_LANES * 3.0
        return -min(total_threat / max_threat, 1.0)

    def _potential_kills(self):
        """击杀潜力"""
        max_kills = getattr(rcfg, 'MAX_KILLS_FOR_NORMALIZATION', 40.0)
        return min(self._weighted_kills, max_kills) / max_kills

    def _potential_strategy(self):
        """策略性潜力: 评估植物位置是否符合策略"""
        phase = self._get_game_phase()
        strategy_score = 0.0
        
        # 按行整理僵尸
        zombies_by_lane = {lane: [] for lane in range(config.N_LANES)}
        for z in self._scene.zombies:
            zombies_by_lane[z.lane].append(z)
        
        # 评估每个植物的位置
        for plant in self._scene.plants:
            ptype = plant.__class__.__name__
            lane = plant.lane
            pos = plant.pos
            lane_has_zombie = len(zombies_by_lane[lane]) > 0
            pressure = self._get_lane_pressure(lane)
            
            if ptype == "Sunflower":
                strategy_score += self._evaluate_sunflower(pos, lane_has_zombie, pressure, phase)
            elif ptype == "Peashooter":
                strategy_score += self._evaluate_peashooter(pos, lane_has_zombie, pressure, phase)
            elif ptype == "Wallnut":
                strategy_score += self._evaluate_wallnut(pos, lane, lane_has_zombie, pressure)
            elif ptype == "Potatomine":
                strategy_score += self._evaluate_potatomine(plant, pos, lane_has_zombie, pressure)
        
        # 归一化
        max_plants = 20
        return strategy_score / max(len(self._scene.plants), 1) * (min(len(self._scene.plants), max_plants) / max_plants)

    def _evaluate_sunflower(self, pos, lane_has_zombie, pressure, phase):
        """评估向日葵位置"""
        score = 0.0
        ideal_pos = getattr(rcfg, 'SUNFLOWER_IDEAL_POS', 3)
        good_bonus = getattr(rcfg, 'SUNFLOWER_GOOD_POS_BONUS', 0.3)
        danger_penalty = getattr(rcfg, 'SUNFLOWER_DANGER_PENALTY', -0.4)
        early_bonus = getattr(rcfg, 'SUNFLOWER_EARLY_GAME_BONUS', 0.5)
        
        # 前期种向日葵奖励
        if phase == 'early':
            score += early_bonus
        
        # 位置在后排（pos < ideal_pos）奖励
        if pos < ideal_pos:
            score += good_bonus
        
        # 在有僵尸的行且位置靠前：危险
        if lane_has_zombie and pos >= 4:
            score += danger_penalty
        
        return score

    def _evaluate_peashooter(self, pos, lane_has_zombie, pressure, phase):
        """评估豌豆射手位置"""
        score = 0.0
        reactive_bonus = getattr(rcfg, 'PEASHOOTER_REACTIVE_BONUS', 0.4)
        premature_penalty = getattr(rcfg, 'PEASHOOTER_PREMATURE_PENALTY', -0.2)
        ideal_min = getattr(rcfg, 'PEASHOOTER_IDEAL_POS_MIN', 2)
        ideal_max = getattr(rcfg, 'PEASHOOTER_IDEAL_POS_MAX', 5)
        
        # 在有僵尸的行部署：响应式防御，好
        if lane_has_zombie:
            score += reactive_bonus
        else:
            # 前期在没僵尸的行提前部署：浪费
            if phase == 'early':
                score += premature_penalty
        
        # 理想位置奖励
        if ideal_min <= pos <= ideal_max:
            score += 0.1
        
        return score

    def _evaluate_wallnut(self, pos, lane, lane_has_zombie, pressure):
        """评估坚果墙位置"""
        score = 0.0
        high_pressure_bonus = getattr(rcfg, 'WALLNUT_HIGH_PRESSURE_BONUS', 0.6)
        low_pressure_penalty = getattr(rcfg, 'WALLNUT_LOW_PRESSURE_PENALTY', -0.3)
        frontline_pos = getattr(rcfg, 'WALLNUT_FRONTLINE_POS', 4)
        frontline_bonus = getattr(rcfg, 'WALLNUT_FRONTLINE_BONUS', 0.4)
        protect_bonus = getattr(rcfg, 'WALLNUT_PROTECT_BONUS', 0.3)
        
        # 高压力时放坚果：好
        if lane_has_zombie and pressure > 0.5:
            score += high_pressure_bonus
        elif not lane_has_zombie:
            # 没僵尸时放坚果：浪费
            score += low_pressure_penalty
        
        # 放在前排：好
        if pos >= frontline_pos:
            score += frontline_bonus
        
        # 检查是否保护了该行其他植物
        plants_behind = sum(1 for p in self._scene.plants 
                          if p.lane == lane and p.pos < pos and p.__class__.__name__ != "Wallnut")
        if plants_behind > 0:
            score += protect_bonus
        
        return score

    def _evaluate_potatomine(self, plant, pos, lane_has_zombie, pressure):
        """评估土豆地雷"""
        score = 0.0
        emergency_bonus = getattr(rcfg, 'POTATOMINE_EMERGENCY_BONUS', 0.8)
        
        # 是否已激活
        is_activated = hasattr(plant, 'attack_cooldown') and plant.attack_cooldown <= 0
        
        # 在高压力行放置土豆：紧急响应
        if lane_has_zombie and pressure > 0.4:
            score += emergency_bonus * pressure
        
        # 已激活的土豆在有僵尸的行：好
        if is_activated and lane_has_zombie:
            score += 0.3
        
        return score

    # ==================== 即时奖励 ====================
    
    def _compute_placement_reward(self, plant_type, lane, pos):
        """计算放置植物的即时奖励"""
        reward = getattr(rcfg, 'PLANT_PLACEMENT_REWARD', 0.1)
        phase = self._get_game_phase()
        
        zombies_in_lane = [z for z in self._scene.zombies if z.lane == lane]
        lane_has_zombie = len(zombies_in_lane) > 0
        pressure = self._get_lane_pressure(lane)
        
        if plant_type == "Sunflower":
            ideal_pos = getattr(rcfg, 'SUNFLOWER_IDEAL_POS', 3)
            if phase == 'early' and pos < ideal_pos:
                reward += getattr(rcfg, 'SUNFLOWER_EARLY_GAME_BONUS', 0.5)
            if lane_has_zombie and pos >= 5:
                reward += getattr(rcfg, 'SUNFLOWER_DANGER_PENALTY', -0.4)
                
        elif plant_type == "Peashooter":
            if lane_has_zombie:
                reward += getattr(rcfg, 'PEASHOOTER_REACTIVE_BONUS', 0.4)
            elif phase == 'early':
                reward += getattr(rcfg, 'PEASHOOTER_PREMATURE_PENALTY', -0.2)
                
        elif plant_type == "Wallnut":
            if lane_has_zombie and pressure > 0.5:
                reward += getattr(rcfg, 'WALLNUT_HIGH_PRESSURE_BONUS', 0.6)
            elif not lane_has_zombie:
                reward += getattr(rcfg, 'WALLNUT_LOW_PRESSURE_PENALTY', -0.3)
            if pos >= getattr(rcfg, 'WALLNUT_FRONTLINE_POS', 4):
                reward += getattr(rcfg, 'WALLNUT_FRONTLINE_BONUS', 0.4)
                
        elif plant_type == "Potatomine":
            if lane_has_zombie and pressure > 0.3:
                reward += getattr(rcfg, 'POTATOMINE_EMERGENCY_BONUS', 0.8)
        
        return reward

    def _check_potatomine_events(self, prev_plants, prev_zombies):
        """检查土豆地雷事件并计算奖励"""
        reward = 0.0
        
        current_plant_ids = {id(p) for p in self._scene.plants}
        current_zombie_ids = {id(z) for z in self._scene.zombies}
        
        prev_potatomines = {id(p): p for p in prev_plants if p.__class__.__name__ == "Potatomine"}
        
        for pid, potato in prev_potatomines.items():
            if pid not in current_plant_ids:
                was_activated = hasattr(potato, 'attack_cooldown') and potato.attack_cooldown <= 0
                was_used = hasattr(potato, 'used') and potato.used == 1
                
                if was_used:
                    reward += getattr(rcfg, 'POTATOMINE_EXPLODE_BONUS', 2.0)
                    
                    if was_activated:
                        reward += getattr(rcfg, 'POTATOMINE_QUICK_EXPLODE_BONUS', 0.5)
                    
                    disappeared_zombies = set(prev_zombies.keys()) - current_zombie_ids
                    same_pos_kills = sum(1 for zid in disappeared_zombies 
                                        if prev_zombies[zid].lane == potato.lane 
                                        and prev_zombies[zid].pos == potato.pos)
                    if same_pos_kills > 1:
                        reward += getattr(rcfg, 'POTATOMINE_MULTI_KILL_BONUS', 1.5) * (same_pos_kills - 1)
                else:
                    reward += getattr(rcfg, 'POTATOMINE_WASTED_PENALTY', -1.5)
        
        return reward

    # ==================== PBRS 奖励计算 ====================
    
    def _compute_shaped_reward(self, base_reward, terminated, truncated, frames_advanced=1):
        """计算总奖励
        
        PBRS 理论: F(s,a,s') = gamma * Phi(s') - Phi(s)
        终止状态: Phi(terminal) = 0
        
        修正: 不再在终止时扣除累积潜力，因为这会导致"策略越好，累积潜力越高，最终惩罚越大"的问题
        改为: 终止时的 PBRS 为 0，让稀疏奖励 (win/lose) 主导终局评估
        """
        gamma = getattr(rcfg, 'GAMMA_PBRS', 0.99)
        
        # 终止状态不计算 PBRS，避免扣除累积潜力
        if terminated or truncated:
            pbrs_reward = 0.0
        else:
            current_potential = self._compute_potential()
            pbrs_reward = gamma * current_potential - self._prev_potential
            self._prev_potential = current_potential
        
        sparse_reward = 0.0
        if terminated:
            sparse_reward = getattr(rcfg, 'REWARD_LOSE', -100.0)
        elif truncated and self._scene.lives > 0:
            sparse_reward = getattr(rcfg, 'REWARD_WIN', 100.0)
        
        if terminated:
            survival_reward = 0.0
        else:
            per_frame = getattr(rcfg, 'REWARD_SURVIVAL_PER_FRAME', 0.5)
            survival_reward = per_frame * max(1, frames_advanced)
        
        # 终止时重置潜力
        if terminated or truncated:
            self._prev_potential = 0.0
        
        total = base_reward + sparse_reward + pbrs_reward + survival_reward + self._strategy_rewards
        self._strategy_rewards = 0.0
        
        return total

    # ==================== 环境接口 ====================
    
    def step(self, action):
        prev_plants = list(self._scene.plants)
        prev_zombies = {id(z): z for z in self._scene.zombies}
        lives_before = self._scene.lives
        chrono_before = self._scene._chrono
        
        action_result = self._take_action(action)
        invalid_penalty = action_result['penalty']
        placement_reward = action_result['placement_reward']
        
        prev_score = self._scene.score
        self._scene.step()
        base_reward = (self._scene.score - prev_score) + invalid_penalty + placement_reward
        
        truncated = self._scene._chrono > config.MAX_FRAMES
        
        while not self._scene.move_available() and not truncated:
            prev_score = self._scene.score
            self._scene.step()
            truncated = self._scene._chrono > config.MAX_FRAMES
            base_reward += self._scene.score - prev_score
        
        frames_advanced = max(1, self._scene._chrono - chrono_before)
        terminated = self._scene.lives <= 0
        
        current_zombies = {id(z): z for z in self._scene.zombies}
        disappeared = set(prev_zombies.keys()) - set(current_zombies.keys())
        
        kills_this_step = 0
        weighted_kills = 0.0
        escaped = 0
        escape_correction = 0.0
        
        kill_values = getattr(rcfg, 'ZOMBIE_KILL_VALUES', {})
        
        for zid in disappeared:
            zombie = prev_zombies[zid]
            if zombie.pos >= 0:
                kills_this_step += 1
                ztype = zombie.__class__.__name__
                kill_value = kill_values.get(ztype, 1.0)
                weighted_kills += kill_value
                # 即时击杀奖励 - 让有击杀的策略明显好于无击杀的
                base_reward += kill_value * getattr(rcfg, 'KILL_REWARD_MULTIPLIER', 2.0)
            else:
                escaped += 1
                if getattr(rcfg, 'CORRECT_ESCAPE_SCORE', True):
                    escape_correction += getattr(zombie, 'SCORE', 0.0)
        
        self._total_kills += kills_this_step
        self._weighted_kills += weighted_kills
        
        if escape_correction:
            base_reward -= escape_correction
        
        life_lost = max(0, lives_before - self._scene.lives)
        if life_lost:
            base_reward += getattr(rcfg, 'LIFE_LOSS_PENALTY', -300.0) * life_lost
        
        potato_reward = self._check_potatomine_events(prev_plants, prev_zombies)
        self._strategy_rewards += potato_reward
        
        shaped_reward = self._compute_shaped_reward(base_reward, terminated, truncated, frames_advanced)
        
        self._reward = shaped_reward
        obs = self._get_obs()
        
        info = {
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "placement_reward": placement_reward,
            "potato_reward": potato_reward,
            "invalid_penalty": invalid_penalty,
            "frames_advanced": frames_advanced,
            "escaped": escaped,
            "life_lost": life_lost,
            "total_kills": self._total_kills,
            "weighted_kills": self._weighted_kills,
            "sun": self._scene.sun,
            "plants": len(self._scene.plants),
            "zombies": len(self._scene.zombies),
            "chrono": self._scene._chrono,
            "game_phase": self._get_game_phase(),
            "phi_sun": self._potential_sun(),
            "phi_defense": self._potential_defense(),
            "phi_threat": self._potential_threat(),
            "phi_kills": self._potential_kills(),
            "phi_strategy": self._potential_strategy(),
            "potential": self._compute_potential(),
        }
        
        return obs, shaped_reward, terminated, truncated, info

    def _get_obs(self):
        obs_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        zombie_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        
        for plant in self._scene.plants:
            obs_grid[plant.lane * config.LANE_LENGTH + plant.pos] = (
                self._plant_no[plant.__class__.__name__] + 1
            )
        for zombie in self._scene.zombies:
            zombie_grid[zombie.lane * config.LANE_LENGTH + zombie.pos] += zombie.hp
        
        action_available = np.array([
            self._scene.plant_cooldowns[name] <= 0
            for name in self.plant_deck
        ])
        action_available *= np.array([
            self._scene.sun >= self.plant_deck[name].COST
            for name in self.plant_deck
        ])
        
        return np.concatenate([
            obs_grid,
            zombie_grid,
            [min(self._scene.sun, MAX_SUN)],
            action_available
        ])

    def reset(self, seed=None, options=None):
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._total_kills = 0
        self._weighted_kills = 0.0
        self._prev_potential = self._compute_potential()
        self._potatomine_tracker = {}
        self._strategy_rewards = 0.0
        return self._get_obs(), {}

    def render(self, mode='human'):
        print(self._scene)
        print(f"Reward: {self._reward:.2f}, Phase: {self._get_game_phase()}")

    def close(self):
        pass

    def _take_action(self, action):
        """执行动作并返回奖励信息"""
        result = {'penalty': 0.0, 'placement_reward': 0.0, 'plant_type': None}
        
        if action > 0:
            action -= 1
            a = action // len(self.plant_deck)
            no_plant = action - len(self.plant_deck) * a
            pos = a // config.N_LANES
            lane = a - pos * config.N_LANES
            
            plant_name = self._plant_names[no_plant]
            plant_type = self._plant_classes[no_plant]
            
            move = Move(plant_name, lane, pos)
            if move.is_valid(self._scene):
                move.apply_move(self._scene)
                result['plant_type'] = plant_type
                result['placement_reward'] = self._compute_placement_reward(plant_type, lane, pos)
            else:
                result['penalty'] = getattr(rcfg, 'INVALID_ACTION_PENALTY', -0.5)
        
        return result

    def mask_available_actions(self):
        empty_cells, available_plants = self._scene.get_available_moves()
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        empty_cells = (empty_cells[0] + config.N_LANES * empty_cells[1]) * len(self.plant_deck)
        for plant in available_plants:
            idx = empty_cells + self._plant_no[plant.__name__] + 1
            mask[idx] = True
        return mask

    def num_observations(self):
        return 2 * config.N_LANES * config.LANE_LENGTH + len(self.plant_deck) + 1
