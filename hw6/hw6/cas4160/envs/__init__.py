from gymnasium.envs.registration import register

def register_envs():
    register(
        id='cheetah-cas4160-v0',
        entry_point='cas4160.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
