"""
纯脚本智能体 - 基于规则的植物大战僵尸AI

策略说明：
1. 第一列（pos=0）全种向日葵，只种向日葵
2. 向日葵和豌豆尽量种在左边
3. 有僵尸出现时，优先种豌豆射手
4. 豌豆射手不够时，考虑土豆地雷（需考虑准备时间14秒）或坚果墙抵挡
5. 评估最有威胁的一路进行决策

使用 position_evaluator 模块中的模拟函数评估威胁程度。
"""

import numpy as np
from pvz import config

# 土豆地雷准备时间（秒）
POTATOMINE_READY_TIME = 14
# 僵尸每秒移动格数（WALKING_SPEED=0.2 格/秒）
ZOMBIE_SPEED = 0.2


class ScriptAgent:
    """基于规则的脚本智能体"""
    
    def __init__(self, n_plants=4):
        """
        初始化脚本智能体
        
        Args:
            n_plants: 植物种类数量（sunflower, peashooter, wall-nut, potatomine）
        """
        self.n_plants = n_plants
        self.n_lanes = config.N_LANES
        self.lane_length = config.LANE_LENGTH
        self.grid_size = self.n_lanes * self.lane_length
        
        # 植物索引 (对应 pvz_env_v2 中的顺序)
        self.SUNFLOWER = 0
        self.PEASHOOTER = 1
        self.WALLNUT = 2
        self.POTATOMINE = 3
        
        # 植物成本
        self.COSTS = {
            self.SUNFLOWER: 50,
            self.PEASHOOTER: 100,
            self.WALLNUT: 50,
            self.POTATOMINE: 25
        }
        
        # 位置配置
        self.SUNFLOWER_COL = 0              # 向日葵只种第一列
        self.PEASHOOTER_PREFERRED_COLS = [1, 2, 3, 4, 5]  # 豌豆射手优先靠左
        self.WALLNUT_PREFERRED_COLS = [6, 7, 5, 4]        # 坚果墙在前方
        self.POTATOMINE_PREFERRED_COLS = [5, 6, 7, 4]     # 土豆地雷在前方
        
        # 状态追踪
        self.frame_count = 0
        
    def _encode_action(self, plant_type, lane, pos):
        """
        将植物类型、行、列编码为动作
        action = 1 + no_plant + n_plants * (lane + n_lanes * pos)
        """
        return 1 + plant_type + self.n_plants * (lane + self.n_lanes * pos)
    
    def _parse_observation(self, obs):
        """
        解析观测向量
        
        观测格式 (pvz_env_v2):
        - obs[0:grid_size]: 植物网格 (0=空, 1=向日葵, 2=豌豆, 3=坚果, 4=土豆)
        - obs[grid_size:2*grid_size]: 僵尸血量网格
        - obs[2*grid_size]: 阳光数量
        - obs[2*grid_size+1:]: 植物是否可用 (冷却完成且阳光足够)
        """
        plant_grid = obs[:self.grid_size].reshape(self.n_lanes, self.lane_length)
        zombie_grid = obs[self.grid_size:2*self.grid_size].reshape(self.n_lanes, self.lane_length)
        sun = obs[2 * self.grid_size]
        plant_available = obs[2 * self.grid_size + 1:]
        
        return {
            'plant_grid': plant_grid,
            'zombie_grid': zombie_grid,
            'sun': sun,
            'plant_available': plant_available
        }
    
    def _count_plants_by_type(self, plant_grid, plant_type):
        """统计某种植物的数量"""
        return np.sum(plant_grid == (plant_type + 1))
    
    def _has_plant_at(self, plant_grid, lane, pos):
        """检查某位置是否有植物"""
        return plant_grid[lane, pos] != 0
    
    def _count_plants_in_lane(self, plant_grid, lane, plant_type):
        """统计某行某种植物的数量"""
        return np.sum(plant_grid[lane] == (plant_type + 1))
    
    def _find_empty_col_in_lane(self, plant_grid, lane, preferred_cols):
        """在指定行找到最佳空位（按优先顺序）"""
        for col in preferred_cols:
            if col < self.lane_length and plant_grid[lane, col] == 0:
                return col
        return None
    
    def _evaluate_lane_threat(self, zombie_grid, plant_grid, lane):
        """
        评估某行的威胁程度
        
        返回: 威胁分数（越高越危险），包含：
        - 僵尸总血量
        - 僵尸位置（越靠左越危险）
        - 该行的防御情况
        
        返回: (threat_score, frontmost_zombie_pos, total_zombie_hp)
        """
        lane_zombies = zombie_grid[lane]
        total_hp = np.sum(lane_zombies)
        
        if total_hp == 0:
            return 0, None, 0
        
        # 找最前方僵尸位置
        frontmost_pos = None
        for pos in range(self.lane_length):
            if lane_zombies[pos] > 0:
                frontmost_pos = pos
                break
        
        # 位置威胁权重：越靠左越危险
        position_threat = 0
        for pos in range(self.lane_length):
            if lane_zombies[pos] > 0:
                # 位置0最危险（权重最高），位置8最不危险
                position_threat += lane_zombies[pos] * (self.lane_length - pos) ** 2
        
        # 防御能力：该行有多少豌豆射手和坚果
        n_peashooters = self._count_plants_in_lane(plant_grid, lane, self.PEASHOOTER)
        n_wallnuts = self._count_plants_in_lane(plant_grid, lane, self.WALLNUT)
        
        # 防御扣分（有防御则威胁降低）
        defense_reduction = n_peashooters * 100 + n_wallnuts * 200
        
        # 最终威胁分 = 总血量 + 位置威胁 - 防御能力
        threat_score = total_hp + position_threat * 0.5 - defense_reduction
        
        return max(0, threat_score), frontmost_pos, total_hp
    
    def _can_potato_reach_in_time(self, frontmost_zombie_pos):
        """
        判断土豆地雷是否能及时准备好
        
        土豆地雷需要14秒准备，僵尸速度约0.2格/秒
        所以僵尸需要至少 14 * 0.2 = 2.8 格的距离才能让土豆准备好
        """
        if frontmost_zombie_pos is None:
            return True
        
        # 估算僵尸到达时间（假设种在僵尸前方2格）
        plant_pos = max(0, frontmost_zombie_pos - 2)
        distance_to_plant = frontmost_zombie_pos - plant_pos
        time_to_reach = distance_to_plant / ZOMBIE_SPEED  # 秒
        
        return time_to_reach >= POTATOMINE_READY_TIME
    
    def _get_all_lanes_by_threat(self, zombie_grid, plant_grid):
        """
        获取所有行按威胁程度排序
        
        返回: [(lane, threat_score, frontmost_pos, total_hp), ...]
        """
        lane_threats = []
        for lane in range(self.n_lanes):
            threat, front_pos, total_hp = self._evaluate_lane_threat(zombie_grid, plant_grid, lane)
            lane_threats.append((lane, threat, front_pos, total_hp))
        
        # 按威胁分数降序排列
        lane_threats.sort(key=lambda x: x[1], reverse=True)
        return lane_threats
    
    def decide_action(self, observation):
        """
        根据当前观测决定动作
        
        策略优先级：
        1. 第一列种向日葵（只在pos=0种向日葵）
        2. 评估威胁最大的行，优先处理
        3. 有僵尸时种豌豆射手
        4. 紧急情况种土豆地雷或坚果墙
        """
        self.frame_count += 1
        parsed = self._parse_observation(observation)
        
        plant_grid = parsed['plant_grid']
        zombie_grid = parsed['zombie_grid']
        sun = parsed['sun']
        plant_available = parsed['plant_available']
        
        # 统计
        n_sunflowers = self._count_plants_by_type(plant_grid, self.SUNFLOWER)
        has_any_zombie = np.sum(zombie_grid) > 0
        
        # 获取按威胁排序的行
        lane_threats = self._get_all_lanes_by_threat(zombie_grid, plant_grid)
        
        # =============== 策略1: 第一列种向日葵 ===============
        # 优先在最安全的行种向日葵（按威胁从低到高）
        if plant_available[self.SUNFLOWER]:
            # 按威胁从低到高找行
            safest_lanes = sorted(lane_threats, key=lambda x: x[1])
            for lane, threat, _, _ in safest_lanes:
                if plant_grid[lane, self.SUNFLOWER_COL] == 0:
                    # 如果有僵尸且该行威胁很高，跳过
                    if threat > 500:
                        continue
                    return self._encode_action(self.SUNFLOWER, lane, self.SUNFLOWER_COL)
        
        # =============== 策略2: 处理有僵尸的行 ===============
        for lane, threat, front_pos, total_hp in lane_threats:
            if total_hp == 0:
                continue  # 这行没僵尸
            
            n_peashooters_in_lane = self._count_plants_in_lane(plant_grid, lane, self.PEASHOOTER)
            n_wallnuts_in_lane = self._count_plants_in_lane(plant_grid, lane, self.WALLNUT)
            
            # ===== 策略2a: 紧急情况处理 =====
            # 如果僵尸已经很靠近（pos <= 3），需要紧急防御
            if front_pos is not None and front_pos <= 3:
                # 紧急：优先坚果墙阻挡
                if plant_available[self.WALLNUT] and n_wallnuts_in_lane == 0:
                    # 在僵尸前方种坚果
                    target_pos = max(0, front_pos - 1)
                    if plant_grid[lane, target_pos] == 0:
                        return self._encode_action(self.WALLNUT, lane, target_pos)
                    # 或者找其他空位
                    pos = self._find_empty_col_in_lane(plant_grid, lane, self.WALLNUT_PREFERRED_COLS)
                    if pos is not None:
                        return self._encode_action(self.WALLNUT, lane, pos)
                
                # 土豆地雷（如果僵尸不是太近）
                if plant_available[self.POTATOMINE] and front_pos >= 2:
                    if self._can_potato_reach_in_time(front_pos):
                        # 在僵尸前方种土豆
                        for try_pos in [max(0, front_pos - 2), max(0, front_pos - 1)]:
                            if plant_grid[lane, try_pos] == 0:
                                return self._encode_action(self.POTATOMINE, lane, try_pos)
            
            # ===== 策略2b: 种豌豆射手 =====
            # 每行至少需要1-2个豌豆射手
            target_peashooters = 2 if total_hp > 200 else 1
            if plant_available[self.PEASHOOTER] and n_peashooters_in_lane < target_peashooters:
                pos = self._find_empty_col_in_lane(plant_grid, lane, self.PEASHOOTER_PREFERRED_COLS)
                if pos is not None:
                    return self._encode_action(self.PEASHOOTER, lane, pos)
            
            # ===== 策略2c: 豌豆不够时考虑土豆地雷 =====
            if not plant_available[self.PEASHOOTER] and n_peashooters_in_lane < 1:
                # 豌豆不可用，考虑土豆地雷
                if plant_available[self.POTATOMINE]:
                    if self._can_potato_reach_in_time(front_pos):
                        pos = self._find_empty_col_in_lane(plant_grid, lane, self.POTATOMINE_PREFERRED_COLS)
                        if pos is not None:
                            return self._encode_action(self.POTATOMINE, lane, pos)
                
                # 土豆也不行，用坚果墙
                if plant_available[self.WALLNUT] and n_wallnuts_in_lane == 0:
                    pos = self._find_empty_col_in_lane(plant_grid, lane, self.WALLNUT_PREFERRED_COLS)
                    if pos is not None:
                        return self._encode_action(self.WALLNUT, lane, pos)
            
            # ===== 策略2d: 加强防御 =====
            # 如果威胁很高但已有豌豆，考虑加坚果墙
            if threat > 300 and n_wallnuts_in_lane == 0 and plant_available[self.WALLNUT]:
                pos = self._find_empty_col_in_lane(plant_grid, lane, self.WALLNUT_PREFERRED_COLS)
                if pos is not None:
                    return self._encode_action(self.WALLNUT, lane, pos)
        
        # =============== 策略3: 预防性布置 ===============
        # 没有紧急威胁时，给每行预先布置豌豆射手
        if plant_available[self.PEASHOOTER]:
            for lane in range(self.n_lanes):
                n_pea = self._count_plants_in_lane(plant_grid, lane, self.PEASHOOTER)
                if n_pea < 1:
                    pos = self._find_empty_col_in_lane(plant_grid, lane, self.PEASHOOTER_PREFERRED_COLS)
                    if pos is not None:
                        return self._encode_action(self.PEASHOOTER, lane, pos)
        
        # =============== 策略4: 继续种向日葵 ===============
        # 如果第一列还有空位，继续种向日葵
        if plant_available[self.SUNFLOWER]:
            for lane in range(self.n_lanes):
                if plant_grid[lane, self.SUNFLOWER_COL] == 0:
                    return self._encode_action(self.SUNFLOWER, lane, self.SUNFLOWER_COL)
        
        # =============== 策略5: 增加攻击力 ===============
        # 每行增加更多豌豆射手
        if plant_available[self.PEASHOOTER]:
            for lane in range(self.n_lanes):
                n_pea = self._count_plants_in_lane(plant_grid, lane, self.PEASHOOTER)
                if n_pea < 3:
                    pos = self._find_empty_col_in_lane(plant_grid, lane, self.PEASHOOTER_PREFERRED_COLS)
                    if pos is not None:
                        return self._encode_action(self.PEASHOOTER, lane, pos)
        
        # =============== 策略6: 预防性坚果墙 ===============
        if plant_available[self.WALLNUT]:
            for lane in range(self.n_lanes):
                n_wall = self._count_plants_in_lane(plant_grid, lane, self.WALLNUT)
                if n_wall == 0:
                    pos = self._find_empty_col_in_lane(plant_grid, lane, self.WALLNUT_PREFERRED_COLS)
                    if pos is not None:
                        return self._encode_action(self.WALLNUT, lane, pos)
        
        # 默认：不操作
        return 0
    
    def reset(self):
        """重置智能体状态"""
        self.frame_count = 0


# ==================== 游戏运行器 ====================
def play_game(agent, env, render=False, max_frames=1000):
    """
    使用指定智能体玩一局游戏
    
    Args:
        agent: 智能体实例
        env: 游戏环境
        render: 是否渲染
        max_frames: 最大帧数
        
    Returns:
        dict: 游戏统计信息
    """
    obs, _ = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()
    
    total_reward = 0
    frame = 0
    
    # 获取底层环境
    unwrapped_env = env.unwrapped
    
    while frame < max_frames:
        if render:
            env.render()
        
        action = agent.decide_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame += 1
        
        if terminated or truncated:
            break
    
    # 判断结果
    is_victory = unwrapped_env._scene.is_victory()
    is_defeat = unwrapped_env._scene.is_defeat()
    
    return {
        'total_reward': total_reward,
        'frames': frame,
        'victory': is_victory,
        'defeat': is_defeat,
        'timeout': not (is_victory or is_defeat)
    }


def evaluate_agent(agent_class, n_episodes=100, render=False):
    """
    评估智能体性能
    
    Args:
        agent_class: 智能体类
        n_episodes: 评估局数
        render: 是否渲染
        
    Returns:
        dict: 评估统计
    """
    import gymnasium as gym
    env = gym.make('gym_pvz:pvz-env-v2')
    agent = agent_class(n_plants=4)
    
    victories = 0
    defeats = 0
    timeouts = 0
    total_rewards = []
    
    for ep in range(n_episodes):
        result = play_game(agent, env, render=render)
        total_rewards.append(result['total_reward'])
        
        if result['victory']:
            victories += 1
        elif result['defeat']:
            defeats += 1
        else:
            timeouts += 1
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: "
                  f"Win={victories}, Lose={defeats}, Timeout={timeouts}, "
                  f"Avg Reward={np.mean(total_rewards):.2f}")
    
    env.close()
    
    return {
        'victories': victories,
        'defeats': defeats,
        'timeouts': timeouts,
        'win_rate': victories / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards)
    }


# ==================== 主程序 ====================
if __name__ == "__main__":
    import gymnasium as gym
    import argparse
    
    parser = argparse.ArgumentParser(description='脚本智能体玩植物大战僵尸')
    parser.add_argument('--render', action='store_true', help='是否渲染游戏画面')
    parser.add_argument('--episodes', type=int, default=10, help='评估局数')
    parser.add_argument('--single', action='store_true', help='只玩一局（详细输出）')
    args = parser.parse_args()
    
    print("=" * 50)
    print("植物大战僵尸 - 脚本智能体")
    print("=" * 50)
    print("策略: 第一列向日葵 + 威胁评估 + 动态防御")
    
    if args.single:
        # 单局游戏
        print("\n开始单局游戏...")
        env = gym.make('gym_pvz:pvz-env-v3')
        agent = ScriptAgent(n_plants=4)
        result = play_game(agent, env, render=True)
        env.close()
        
        print("\n" + "=" * 50)
        print("游戏结果:")
        print(f"  总奖励: {result['total_reward']:.2f}")
        print(f"  帧数: {result['frames']}")
        print(f"  结果: {'胜利' if result['victory'] else '失败' if result['defeat'] else '超时'}")
    else:
        # 评估模式
        print(f"\n开始评估 {args.episodes} 局...")
        stats = evaluate_agent(ScriptAgent, n_episodes=args.episodes, render=args.render)
        
        print("\n" + "=" * 50)
        print("评估结果:")
        print(f"  胜利: {stats['victories']}/{args.episodes} ({stats['win_rate']*100:.1f}%)")
        print(f"  失败: {stats['defeats']}/{args.episodes}")
        print(f"  超时: {stats['timeouts']}/{args.episodes}")
        print(f"  平均奖励: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
