from gymnasium.envs.registration import register, registry

def safe_register(id, entry_point):
    if id not in registry:
        register(id=id, entry_point=entry_point)

safe_register(
    id='pvz-env-v0',
    entry_point='gym_pvz.envs:PVZEnv'
)

safe_register(
    id='pvz-env-v1',
    entry_point='gym_pvz.envs:PVZEnv_V1'
)

safe_register(
    id='pvz-env-v01',
    entry_point='gym_pvz.envs:PVZEnv_V01'
)

safe_register(
    id='pvz-env-v2',
    entry_point='gym_pvz.envs:PVZEnv_V2'
)
