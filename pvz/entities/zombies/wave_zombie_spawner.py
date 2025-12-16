from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from ... import config
import random

INITIAL_OFFSET = 10
SPAWN_INTERVAL = 8
MAX_WAVES = 3  # 最大波数，第MAX_WAVES波后不再生成新僵尸

class WaveZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._timer = INITIAL_OFFSET * config.FPS - 1
        self._wave_timer= 10*INITIAL_OFFSET * config.FPS - 1
        self.p=0.05
        self._wave_count = 0  # 当前已生成的大波数
        self._finished = False  # 是否已完成所有波次
        
    def is_finished(self):
        """返回是否已完成所有僵尸的生成"""
        return self._finished
        
    def spawn(self, scene):
        # 如果已完成所有波次，不再生成
        if self._finished:
            return
            
        if self._timer <= 0 and self._wave_timer>0 :
            lane = random.choice(range(config.N_LANES))
            s=random.random()
            if(s<self.p):
                scene.add_zombie(Zombie_bucket(lane))
            elif(s<3*self.p):
                scene.add_zombie(Zombie_cone(lane))
            else:
                scene.add_zombie(Zombie(lane))
            self._timer = SPAWN_INTERVAL * config.FPS - 1
        else:
            if(self._wave_timer>0):
                self._timer -= 1
                self._wave_timer -=1
            else:
                # 大波来袭
                self._wave_count += 1
                scene.add_zombie(Zombie_flag(0))
                for lane in range(config.N_LANES):
                    s=random.random()
                    if(s<self.p):
                        scene.add_zombie(Zombie_bucket(lane))
                    elif(s<3*self.p):
                        scene.add_zombie(Zombie_cone(lane))
                    else:
                        scene.add_zombie(Zombie(lane))
                
                # 检查是否已达到最大波数
                if self._wave_count >= MAX_WAVES:
                    self._finished = True
                else:
                    self._wave_timer = 20 * SPAWN_INTERVAL * config.FPS - 1
                    self._timer = 10 * INITIAL_OFFSET * config.FPS - 1
                    self.p=min(self.p*2,1)