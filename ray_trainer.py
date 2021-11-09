import ray
# https://docs.ray.io/en/latest/rllib-env.html
from ray.rllib.agents import ppo

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

class DeliveryEnvRLlib(DeliveryEnv):
    # RLlib compatible version of DeliveryEnv
    def __init__(self, env_config):
        init_state = env_config['init_state']
        super().__init__(init_state)

# Setup env
init_state = DeliveryState(5, 4)
init_state.player = (2, 0)
init_state.spawners[4, 2] = 1
init_state.dropoffs[1, 3] = 1
init_state.dropoffs[0, 0] = 1
#env = DeliveryEnv(init_state)

ray.init()
# trainer = ppo.PPOTrainer(env=DeliveryEnvRLlib, config={
#     "env_config": {'init_state': init_state},  # config to pass to env class
# })
# while True:
#     print(trainer.train())

'''

'''

ray.tune.run(ppo.PPOTrainer, config={"env": DeliveryEnvRLlib, "env_config": {'init_state': init_state}})

