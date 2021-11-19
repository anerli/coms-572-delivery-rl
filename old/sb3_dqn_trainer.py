from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv
from delivery_action import DeliveryAction

import torch

import sys

if len(sys.argv) == 2 and sys.argv[1] == '-n':
    load = False
else:
    load = True

init_state = DeliveryState(4, 4, 2, 1)
env = DeliveryEnv(init_state)

if load:
    model = DQN.load('dqn_delivery', env=env)
else:
    # These kwargs don't work for DQN
    policy_kwargs = dict(
        #activation_fn=torch.nn.ReLU,
        #net_arch=[dict(pi=[32, 32], vf=[32, 32])]
        net_arch=[32, 32]
    )
    model = DQN('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)

try:
    while True:
        model.learn(total_timesteps=int(2e5))
        model.save('dqn_delivery')
        print('Saved Model')
except KeyboardInterrupt:
    pass

model.save('dqn_delivery')