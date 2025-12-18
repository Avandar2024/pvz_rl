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

# 僵尸生成概率配置（按波次递增难度）
# wave_count = 0: 第一波之前（只有普通僵尸）
# wave_count = 1: 第一波结束后（开始出现路障）
# wave_count = 2: 第二波结束后（开始出现铁桶）
ZOMBIE_SPAWN_CONFIG = {
    0: {'bucket_prob': 0.0, 'cone_prob': 0.0},     # 第一波前：只有普通僵尸
    1: {'bucket_prob': 0.0, 'cone_prob': 0.06},    # 第二波前：路障6%
    2: {'bucket_prob': 0.02, 'cone_prob': 0.10},   # 第三波前：铁桶2%，路障10%
    3: {'bucket_prob': 0.05, 'cone_prob': 0.15},   # 第三波后：铁桶5%，路障15%
}

class WaveZombieSpawner(ZombieSpawner):

    def __init__(self):
        self._timer = INITIAL_OFFSET * config.FPS - 1
        self._wave_timer= 10*INITIAL_OFFSET * config.FPS - 1
        self._wave_count = 0  # 当前已生成的大波数
        self._finished = False  # 是否已完成所有波次
        
    def is_finished(self):
        """返回是否已完成所有僵尸的生成"""
        return self._finished
    
    def _get_spawn_probs(self):
        """根据当前波次获取僵尸生成概率"""
        wave = min(self._wave_count, max(ZOMBIE_SPAWN_CONFIG.keys()))
        return ZOMBIE_SPAWN_CONFIG[wave]
    
    def _spawn_zombie_by_wave(self, scene, lane):
        """根据当前波次概率生成对应类型的僵尸"""
        probs = self._get_spawn_probs()
        s = random.random()
        
        if s < probs['bucket_prob']:
            scene.add_zombie(Zombie_bucket(lane))
        elif s < probs['bucket_prob'] + probs['cone_prob']:
            scene.add_zombie(Zombie_cone(lane))
        else:
            scene.add_zombie(Zombie(lane))
        
    def spawn(self, scene):
        # 如果已完成所有波次，不再生成
        if self._finished:
            return
            
        if self._timer <= 0 and self._wave_timer > 0:
            lane = random.choice(range(config.N_LANES))
            self._spawn_zombie_by_wave(scene, lane)
            self._timer = SPAWN_INTERVAL * config.FPS - 1
        else:
            if self._wave_timer > 0:
                self._timer -= 1
                self._wave_timer -= 1
            else:
                # 大波来袭
                self._wave_count += 1
                scene.add_zombie(Zombie_flag(0))
                for lane in range(config.N_LANES):
                    self._spawn_zombie_by_wave(scene, lane)
                
                # 检查是否已达到最大波数
                if self._wave_count >= MAX_WAVES:
                    self._finished = True
                else:
                    self._wave_timer = 20 * SPAWN_INTERVAL * config.FPS - 1
                    self._timer = 10 * INITIAL_OFFSET * config.FPS - 1