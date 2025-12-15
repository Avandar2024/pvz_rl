import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, MultiBinary, Tuple, Discrete
from pvz import Scene, BasicZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut
import numpy as np

MAX_ZOMBIE_PER_CELL = 10
MAX_SUN = 10000


class PVZEnv_V1(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.plant_deck = {"sunflower": Sunflower, "peashooter": Peashooter, "wall-nut": Wallnut}

        self.action_space = Discrete(len(self.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)
        # self.action_space = MultiDiscrete([len(self.plant_deck), config.N_LANES, config.LANE_LENGTH]) # plant, lane, pos
        self.observation_space = Tuple(
            [MultiDiscrete([len(self.plant_deck) + 1] * (config.N_LANES * config.LANE_LENGTH)),
             MultiDiscrete([MAX_ZOMBIE_PER_CELL + 1] * (config.N_LANES * config.LANE_LENGTH)),
             Discrete(MAX_SUN),
             MultiBinary(len(self.plant_deck))])  # Action available

        "Which plant on the cell, is the lane attacked, is there a mower on the lane"
        self._plant_names = [plant_name for plant_name in self.plant_deck]
        self._plant_classes = [self.plant_deck[plant_name].__name__ for plant_name in self.plant_deck]
        self._plant_no = {self._plant_classes[i]: i for i in range(len(self._plant_names))}
        self._scene = Scene(self.plant_deck, BasicZombieSpawner())
        self._reward = 0

    def step(self, action):
        """
        Gymnasium step() new API:
        returns obs, reward, terminated, truncated, info
        """
        # Apply action
        self._take_action(action)
        # Minimum one scene step
        self._scene.step()
        reward = self._scene.score
        # Continue stepping while no further action can be taken
        terminated = self._scene.is_defeat() or self._scene.is_victory()
        while (not self._scene.move_available()) and (not terminated):
            self._scene.step()
            reward += self._scene.score
            terminated = self._scene.is_defeat() or self._scene.is_victory()
        # Read observation
        obs = self._get_obs()
        # Episode termination conditions
        truncated = False  # 胜利/失败由 is_victory/is_defeat 判定
        # Keep score for debug render()
        self._reward = reward
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        obs_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        zombie_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        for plant in self._scene.plants:
            obs_grid[plant.lane * config.LANE_LENGTH + plant.pos] = self._plant_no[plant.__class__.__name__] + 1
        for zombie in self._scene.zombies:
            zombie_grid[zombie.lane * config.LANE_LENGTH + zombie.pos] += 1
        action_available = np.array([self._scene.plant_cooldowns[plant_name] <= 0 for plant_name in self.plant_deck])
        action_available *= np.array(
            [self._scene.sun >= self.plant_deck[plant_name].COST for plant_name in self.plant_deck])
        return np.concatenate([obs_grid, zombie_grid, [self._scene.sun], action_available])

    def reset(self, seed=None, options=None):
        """Reset the environment (accepts seed/options for Gymnasium compatibility)."""
        # seed/options are accepted for compatibility but not used here.
        self._scene = Scene(self.plant_deck, BasicZombieSpawner())
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

    def num_observations(self):
        return 2 * config.N_LANES * config.LANE_LENGTH + len(self.plant_deck) + 1
