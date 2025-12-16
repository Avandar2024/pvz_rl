from ... import config
from .zombie import Zombie

# 导入击杀奖励配置
try:
    from ... import reward_config
    _BUCKET_KILL_REWARD = reward_config.ZOMBIE_BUCKET_KILL_REWARD
except (ImportError, AttributeError):
    _BUCKET_KILL_REWARD = 120  # 默认值

class Zombie_bucket(Zombie):

    MAX_HP = 1100
    SCORE = _BUCKET_KILL_REWARD  # 击杀铁桶僵尸奖励

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
        if self.hp<190:
            zombie = Zombie(self.lane,self.pos)
            zombie.hp = self.hp
            zombie._offset = self._offset
            scene.add_zombie(zombie)
            self.hp=0
