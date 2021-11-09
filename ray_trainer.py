import ray
# https://docs.ray.io/en/latest/rllib-env.html
from ray.rllib.agents import ppo

from ray.tune.logger import LoggerCallback, JsonLoggerCallback
import os
import json
from typing import Dict, List

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

class DeliveryEnvRLlib(DeliveryEnv):
    # RLlib compatible version of DeliveryEnv
    def __init__(self, env_config):
        init_state = env_config['init_state']
        super().__init__(init_state)

# Setup env
init_state = DeliveryState(5, 4, (2, 0))
init_state.spawners[4, 2] = 1
init_state.dropoffs[1, 3] = 1
init_state.dropoffs[0, 0] = 1
#env = DeliveryEnv(init_state)

# FIXME: Does not want to give me these resources:
# Resources requested: 3.0/5 CPUs, 0/1 GPUs, 0.0/10.48 GiB heap, 0.0/5.24 GiB objects
ray.init(num_cpus=5, num_gpus=1)
# trainer = ppo.PPOTrainer(env=DeliveryEnvRLlib, config={
#     "env_config": {'init_state': init_state},  # config to pass to env class
# })
# while True:
#     print(trainer.train())

# class CustomLoggerCallback(LoggerCallback):
#     """Custom logger interface"""

#     def __init__(self, filename: str = "log.txt"):
#         self._trial_files = {}
#         self._filename = filename

#     def log_trial_start(self, trial):
#         trial_logfile = os.path.join(trial.logdir, self._filename)
#         self._trial_files[trial] = open(trial_logfile, "at")

#     def log_trial_result(self, iteration: int, trial, result: Dict):
#         if trial in self._trial_files:
#             self._trial_files[trial].write(json.dumps(result))

#     def on_trial_complete(self, iteration: int, trials, trial, **info):
#         if trial in self._trial_files:
#             self._trial_files[trial].close()
#             del self._trial_files[trial]

'''
agent seems to be having trouble gaining any reward at all.
may have to add a kind of heursitic to push them to start making deliveries.
'''
# https://docs.ray.io/en/latest/train/api.html
# https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
# trainer = PPOTrainer(env="CartPole-v0", config={"framework": "tf2", "num_workers": 0})
ray.tune.run(
    ppo.PPOTrainer,
    config={
        "env": DeliveryEnvRLlib,
        "env_config": {'init_state': init_state},
        #"workers": 4,
        #"use_gpu": True,
    },
    #resources_per_trial={"cpu": 5, "gpu": 1},
    
    checkpoint_freq=10,
    keep_checkpoints_num=5,
    name="delivery_ray_experiment", # need name so it knows where to save / retrieve checkpoints if resuming
    resume=False,
    callbacks=[JsonLoggerCallback()]#[CustomLoggerCallback("log_test.txt")]
)

'''
ValueError: Resources for <class 'ray.rllib.agents.trainer_template.PPO'> have been automatically set to <ray.tune.utils.placement_groups.PlacementGroupFactory object at 0x0000021C08D4E1F0> by its `default_resource_request()` method. Please clear the `resources_per_trial` option.
'''