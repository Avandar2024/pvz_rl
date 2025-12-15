import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete
from pvz import Scene, WaveZombieSpawner, Move, config, Sunflower, Peashooter, Wallnut, Potatomine
import numpy as np

MAX_ZOMBIE_HP = 10000
MAX_SUN = 10000
MAX_COOLDOWN = 20  # Potatomine/Wallnut


class PVZEnv_V2(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.plant_deck = {"sunflower": Sunflower, "peashooter": Peashooter, "wall-nut": Wallnut,
                           "potatomine": Potatomine}

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

    def step(self, action):
        """
        New Gymnasium step API:
        return obs, reward, terminated, truncated, info
        """
        # Apply action
        self._take_action(action)
        prev_score = self._scene.score
        self._scene.step()  # Minimum one step
        reward = self._scene.score - prev_score
        # Check if episode ended by time limit
        truncated = self._scene._chrono > config.MAX_FRAMES
        # Continue stepping until another move is available
        while (not self._scene.move_available()) and (not truncated):
            prev_score = self._scene.score
            self._scene.step()
            truncated = self._scene._chrono > config.MAX_FRAMES
            reward += self._scene.score - prev_score
        # Observation
        obs = self._get_obs()
        # Episode ends if lives run out OR time limit reached
        terminated = self._scene.lives <= 0
        # Save reward for rendering
        self._reward = reward
        return obs, reward, terminated, truncated, {}

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
