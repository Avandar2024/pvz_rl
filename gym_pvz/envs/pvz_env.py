import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
from pvz import Scene, BasicZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut
import numpy as np


class PVZEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.plant_deck = {
            "sunflower": Sunflower,
            "peashooter": Peashooter,
            "wall-nut": Wallnut
        }

        # Action space: choosing plant + (lane, pos) + no-op
        self.action_space = Discrete(len(self.plant_deck) * config.N_LANES * config.LANE_LENGTH + 1)

        # Observation space
        grid_size = config.N_LANES * config.LANE_LENGTH
        nvec = [len(self.plant_deck) + 1] * grid_size + [2] * config.N_LANES + [2] * config.N_LANES
        self.observation_space = MultiDiscrete(nvec)

        self._plant_names = list(self.plant_deck.keys())
        self._plant_classes = [cls.__name__ for cls in self.plant_deck.values()]
        self._plant_no = {cls.__name__: i for i, cls in enumerate(self.plant_deck.values())}

        self._scene = Scene(self.plant_deck, BasicZombieSpawner())
        self._reward = 0

    # ----------------------
    # Gymnasium step() new API
    # ----------------------
    def step(self, action):

        self._take_action(action)
        self._scene.step()  # At least one scene step
        reward = self._scene.score

        # Continue stepping until action is available again
        while not self._scene.move_available():
            self._scene.step()
            reward += self._scene.score

        obs = self._get_obs()
        terminated = self._scene.lives <= 0
        truncated = False  # No truncation logic in PVZ

        self._reward = reward

        return obs, reward, terminated, truncated, {}

    # ----------------------
    # Gymnasium reset() new API
    # ----------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._scene = Scene(self.plant_deck, BasicZombieSpawner())
        obs = self._get_obs()
        return obs, {}

    # ----------------------
    # Rendering
    # ----------------------
    def render(self):
        print(self._scene)
        print("Score since last action:", self._reward)

    def close(self):
        pass

    # ----------------------
    # Internal helper methods
    # ----------------------
    def _take_action(self, action):
        if action > 0:  # action 0 = no-op
            action -= 1
            a = action // len(self.plant_deck)
            no_plant = action % len(self.plant_deck)
            pos = a // config.N_LANES
            lane = a % config.N_LANES

            move = Move(self._plant_names[no_plant], lane, pos)
            if move.is_valid(self._scene):
                move.apply_move(self._scene)

    def _get_obs(self):
        obs_grid = np.zeros(config.N_LANES * config.LANE_LENGTH, dtype=int)
        for plant in self._scene.plants:
            obs_grid[plant.lane * config.LANE_LENGTH + plant.pos] = \
                self._plant_no[plant.__class__.__name__] + 1

        return np.concatenate([
            obs_grid,
            self._scene.grid._lanes.astype(int),
            self._scene.grid._mowers.astype(int)
        ])

    def num_observations(self):
        return config.N_LANES * (config.LANE_LENGTH + 2)
