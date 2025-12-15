import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
from pvz import Scene, WaveZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut, Potatomine
import numpy as np

# 观测空间上限 (不常调整)
MAX_ZOMBIE_HP = 10000
MAX_SUN = 10000
MAX_COOLDOWN = 20  # Potatomine/Wallnut

# ==================== 奖励塑形参数从 config 导入 ====================
# 所有可调参数集中在 pvz/config.py 中，便于统一管理


class PVZEnv_V2(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.plant_deck = {"sunflower": Sunflower, "peashooter": Peashooter, "wall-nut": Wallnut,
                           "potatomine": Potatomine}

        # 用于 PBRS 的状态跟踪
        self._prev_potential = 0.0
        self._total_kills = 0
        self._weighted_kills = 0.0  # 按僵尸类型加权的击杀分
        self._prev_zombie_ids = set()  # 用于精确追踪击杀 vs 进家

        self.action_space = Discrete(len(self.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)
        # self.action_space = MultiDiscrete([len(self.plant_deck), config.N_LANES, config.LANE_LENGTH]) # plant, lane, pos
        # The environment returns a flattened observation vector (concatenation of
        # plant-grid, zombie-grid, sun, and action_available). Declare a single
        # MultiDiscrete observation_space with matching nvec for each entry.
        grid_size = config.N_LANES * config.LANE_LENGTH
        nvec = [len(self.plant_deck) + 1] * grid_size + [MAX_ZOMBIE_HP] * grid_size + [MAX_SUN] + [2] * len(
            self.plant_deck)
        self.observation_space = MultiDiscrete(nvec)

        "Which plant on the cell, is the lane attacked, is there a mower on the lane"
        self._plant_names = [plant_name for plant_name in self.plant_deck]
        self._plant_classes = [self.plant_deck[plant_name].__name__ for plant_name in self.plant_deck]
        self._plant_no = {self._plant_classes[i]: i for i in range(len(self._plant_names))}
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())
        self._reward = 0

        # 植物类型权重 (从 config 导入)
        self._plant_defense_value = config.PLANT_DEFENSE_VALUES

        # 僵尸类型威胁权重 (从 config 导入)
        self._zombie_threat_weight = config.ZOMBIE_THREAT_WEIGHTS

    # ==================== 潜力函数实现 ====================

    def _compute_potential(self):
        """
        计算当前状态的潜力函数 Φ(s)
        Φ(s) = w1*Φ_sun + w2*Φ_defense + w3*Φ_threat + w4*Φ_kills
        """
        phi_sun = self._potential_sun()
        phi_defense = self._potential_defense()
        phi_threat = self._potential_threat()
        phi_kills = self._potential_kills()

        potential = (
            config.W_SUN * phi_sun +
            config.W_DEFENSE * phi_defense +
            config.W_THREAT * phi_threat +
            config.W_KILLS * phi_kills
        )
        return potential

    def _potential_sun(self):
        """
        资源管理潜力 Φ_sun(s)
        包括: 当前阳光 + 向日葵数量带来的未来产出能力
        """
        # 当前阳光归一化
        current_sun_normalized = min(self._scene.sun, config.MAX_SUN_CAPACITY) / config.MAX_SUN_CAPACITY

        # 向日葵数量 -> 未来阳光产出能力
        sunflower_count = sum(1 for p in self._scene.plants if p.__class__.__name__ == "Sunflower")
        sunflower_bonus = sunflower_count * 0.1  # 每个向日葵加 0.1 的潜力

        return current_sun_normalized + sunflower_bonus

    def _potential_defense(self):
        """
        防御覆盖潜力 Φ_defense(s)
        每行的植物覆盖情况 × 植物防御价值
        """
        defense_score = 0.0

        for lane in range(config.N_LANES):
            lane_plants = [p for p in self._scene.plants if p.lane == lane]
            if not lane_plants:
                continue

            # 计算该行的防御价值
            lane_defense = sum(
                self._plant_defense_value.get(p.__class__.__name__, 1.0)
                for p in lane_plants
            )

            # 考虑植物位置：越靠前(pos越小)防御价值越高
            position_bonus = sum(
                (config.LANE_LENGTH - p.pos) / config.LANE_LENGTH * 0.5
                for p in lane_plants
            )

            defense_score += lane_defense + position_bonus

        # 归一化 (假设最大防御分约为每行3个高价值植物)
        max_defense = config.N_LANES * 3 * 2.5
        return defense_score / max_defense

    def _potential_threat(self):
        """
        威胁降低潜力 Φ_threat(s)
        僵尸越靠近房子，威胁越大（负潜力）
        使用指数惩罚: -Σ weight_j * exp(-decay * x_j)
        """
        threat = 0.0

        for zombie in self._scene.zombies:
            # 僵尸位置 (pos=0 是最左边/房子, pos=8 是最右边/出生点)
            # 实际位置还要考虑 offset
            actual_pos = zombie.pos + zombie.get_offset()

            # 获取僵尸类型权重
            zombie_type = zombie.__class__.__name__
            weight = self._zombie_threat_weight.get(zombie_type, 1.0)

            # 考虑僵尸血量 (血量越高威胁越大)
            hp_factor = zombie.hp / zombie.MAX_HP

            # 指数惩罚：僵尸越靠近房子(pos越小)，惩罚越大
            # exp(-decay * pos) 当 pos=0 时最大，pos=8 时最小
            threat += weight * hp_factor * np.exp(-config.THREAT_DECAY * actual_pos)

        # 返回负值 (威胁越大，潜力越低)
        # 归一化系数：假设最坏情况是5只满血铁桶僵尸在 pos=0
        max_threat = 5 * 2.5 * 1.0 * np.exp(0)
        return -threat / max_threat

    def _potential_kills(self):
        """
        消灭进度潜力 Φ_kills(s)
        按僵尸类型加权的击杀分 (铁桶僵尸价值更高)
        """
        # 归一化：假设一局最多约 100 分加权击杀
        return min(self._weighted_kills, 100.0) / 100.0

    def _compute_shaped_reward(self, base_reward, terminated, truncated):
        """
        计算塑形后的总奖励
        r_shaped = r_base + r_sparse + F_pbrs + r_survival

        其中 F_pbrs = γ * Φ(s') - Φ(s) (Potential-Based Reward Shaping)
        """
        # 1. 计算当前状态潜力
        current_potential = self._compute_potential()

        # 2. PBRS 塑形奖励: F = γΦ(s') - Φ(s)
        # 注意：在终止状态，Φ(s') = 0
        if terminated or truncated:
            pbrs_reward = -self._prev_potential  # γ * 0 - Φ(s)
        else:
            pbrs_reward = config.GAMMA_PBRS * current_potential - self._prev_potential

        # 3. 稀疏的大奖励
        sparse_reward = 0.0
        if terminated:
            sparse_reward = config.REWARD_LOSE  # 僵尸进家了
        elif truncated and self._scene.lives > 0:
            sparse_reward = config.REWARD_WIN   # 撑过了所有帧数，胜利

        # 4. 存活奖励 (每步小正奖励)
        survival_reward = config.REWARD_SURVIVAL_PER_FRAME if not terminated else 0.0

        # 5. 更新前一状态潜力 (用于下一步计算)
        self._prev_potential = current_potential if not (terminated or truncated) else 0.0

        # 总奖励
        total_reward = base_reward + sparse_reward + pbrs_reward + survival_reward

        return total_reward

    def step(self, action):
        """
        New Gymnasium step API:
        return obs, reward, terminated, truncated, info

        奖励包含:
        1. base_reward: 来自 Scene.score 的原始奖励 (击杀等)
        2. sparse_reward: 胜利/失败的大奖励
        3. pbrs_reward: 基于潜力函数的塑形奖励 F = γΦ(s') - Φ(s)
        4. survival_reward: 每步存活的小奖励
        """
        # 记录动作前的僵尸集合 (用于精确计算击杀)
        # 使用 id() 追踪每只僵尸对象
        prev_zombies = {id(z): z for z in self._scene.zombies}

        # Apply action
        self._take_action(action)
        prev_score = self._scene.score
        self._scene.step()  # Minimum one step
        reward = self._scene.score
        base_reward = self._scene.score - prev_score

        # Check if episode ended by time limit
        truncated = self._scene._chrono > config.MAX_FRAMES

        # Continue stepping until another move is available
        terminated = self._scene.is_defeat() or self._scene.is_victory()
        truncated = False  # 胜利/失败由 is_victory/is_defeat 判定
        while (not self._scene.move_available()) and (not terminated):
        while (not self._scene.move_available()) and (not truncated):
            prev_score = self._scene.score
            self._scene.step()
            reward += self._scene.score
            terminated = self._scene.is_defeat() or self._scene.is_victory()
            truncated = self._scene._chrono > config.MAX_FRAMES
            base_reward += self._scene.score - prev_score

        # Observation
        obs = self._get_obs()
        # 终局：场上僵尸被清空（胜利）或基地被攻破（失败）

        # Episode ends if lives run out
        terminated = self._scene.lives <= 0

        # 更新击杀计数 (区分真正击杀 vs 进家)
        current_zombies = {id(z): z for z in self._scene.zombies}

        # 找出消失的僵尸
        disappeared_ids = set(prev_zombies.keys()) - set(current_zombies.keys())

        kills_this_step = 0
        weighted_kills_this_step = 0.0

        for zid in disappeared_ids:
            zombie = prev_zombies[zid]
            # 进家的僵尸 pos 会变成 -1 后被移除
            # 被击杀的僵尸 hp 会变成 <=0 后被移除，但 pos >= 0
            # 通过检查 zombie.pos 来区分
            if zombie.pos >= 0:
                # 被击杀 (hp <= 0 时移除，但 pos 还在场内)
                kills_this_step += 1
                zombie_type = zombie.__class__.__name__
                kill_value = config.ZOMBIE_KILL_VALUES.get(zombie_type, 1.0)
                weighted_kills_this_step += kill_value
            # else: 进家了 (pos < 0)，不计入击杀

        self._total_kills += kills_this_step
        self._weighted_kills += weighted_kills_this_step

        # 计算塑形后的总奖励
        shaped_reward = self._compute_shaped_reward(base_reward, terminated, truncated)

        # Save reward for rendering
        self._reward = shaped_reward

        # info 中包含诊断信息 (包括各潜力函数分量，便于调试)
        info = {
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "total_kills": self._total_kills,
            "weighted_kills": self._weighted_kills,
            "sun": self._scene.sun,
            "plants": len(self._scene.plants),
            "zombies": len(self._scene.zombies),
            "chrono": self._scene._chrono,
            # 潜力函数诊断
            "phi_sun": self._potential_sun(),
            "phi_defense": self._potential_defense(),
            "phi_threat": self._potential_threat(),
            "phi_kills": self._potential_kills(),
            "potential": self._compute_potential(),
        }

        return obs, shaped_reward, terminated, truncated, info

    def _get_obs(self):
        obs_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        zombie_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        for plant in self._scene.plants:
            obs_grid[plant.lane * config.LANE_LENGTH + plant.pos] = self._plant_no[plant.__class__.__name__] + 1
        for zombie in self._scene.zombies:
            zombie_grid[zombie.lane * config.LANE_LENGTH + zombie.pos] += zombie.hp
        action_available = np.array([self._scene.plant_cooldowns[plant_name] <= 0 for plant_name in self.plant_deck])
        action_available *= np.array(
            [self._scene.sun >= self.plant_deck[plant_name].COST for plant_name in self.plant_deck])
        return np.concatenate([obs_grid, zombie_grid, [min(self._scene.sun, MAX_SUN)], action_available])

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Accepts the newer Gymnasium reset signature (seed, options) for
        compatibility. Returns the observation (old-style single return),
        which is still accepted by Gymnasium's passive checker.
        """
        # Note: seed/options are accepted for compatibility but not used here.
        self._scene = Scene(self.plant_deck, WaveZombieSpawner())

        # 重置 PBRS 状态
        self._total_kills = 0
        self._weighted_kills = 0.0
        self._prev_potential = self._compute_potential()  # 初始状态的潜力

        obs = self._get_obs()
        return obs, {}

    def render(self, mode='human'):
        print(self._scene)
        print("Score since last action: " + str(self._reward))

    def close(self):
        pass

    def _take_action(self, action):
        if action > 0:  # action = 0 : no action
            # action = no_plant + n_plants * (lane + n_lanes * pos)
            action -= 1
            a = action // len(self.plant_deck)
            no_plant = action - len(self.plant_deck) * a
            pos = a // config.N_LANES
            lane = a - pos * config.N_LANES
            move = Move(self._plant_names[no_plant], lane, pos)
            if move.is_valid(self._scene):
                move.apply_move(self._scene)
            # else:
            #     print("made a wrong move??")
            #     input()

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
