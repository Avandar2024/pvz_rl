from ..entity import Entity

from ... import config

# 导入击杀奖励配置
try:
    from ... import reward_config
    _ZOMBIE_KILL_REWARD = reward_config.ZOMBIE_KILL_REWARD
except (ImportError, AttributeError):
    _ZOMBIE_KILL_REWARD = 120  # 默认值

class Zombie(Entity):

    MAX_HP = 190  # 190 normally
    WALKING_SPEED = 5 # Speed per square
    ATTACK_SPEED = 100 # Damage per second
    SCORE = _ZOMBIE_KILL_REWARD  # 击杀奖励（从 reward_config 导入）

    def __init__(self, lane, pos=config.LANE_LENGTH - 1): # Zombie starts at the very right
        """
        lane: lane where the zombie is
        """

        super().__init__(lane)
        self.pos = pos
        self._attack = self.ATTACK_SPEED // config.FPS
        self._cell_length = self.WALKING_SPEED * config.FPS # How many frames to pass a cell
        self._offset = self._cell_length - 1 # Offset used to model the displacement of the zombie on its cell


    def step(self, scene):
        if scene.grid.is_empty(self.lane, self.pos):
            if self._offset <= 0:
                self.pos -= 1
                self._offset = self._cell_length - 1
                if self.pos < 0: # If the zombie reached the house, we lose a life and the zombie disappear
                    scene.zombie_reach(self.lane)
                    self.hp = 0
            else:
                self._offset -= 1
        else:
            for plant in scene.plants:
                if (plant.lane == self.lane) and (plant.pos == self.pos):
                    self.attack(plant)
                    break

    def attack(self, plant):
        plant.hp -= self._attack

    def get_offset(self):
        return self._offset/self._cell_length

    def __str__(self):
        return ("Lane: " + str(self.lane) + " Pos: " + str(self.pos)
                + " Health: " + str(self.hp))