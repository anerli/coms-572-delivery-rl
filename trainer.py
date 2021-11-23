from stable_baselines3 import DQN

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

from tester import test

import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-l', '--loadfile', type=str, required=False)
parser.add_argument('-s', '--savefile', type=str, required=False)
parser.add_argument('-e', '--steplimit', type=int, default=60, required=False)

args = parser.parse_args()

init_state = DeliveryState(4, 4, 2, 1)
if args.steplimit:
    init_state.step_lim = args.steplimit
env = DeliveryEnv(init_state)

if args.loadfile:
    model = DQN.load(args.loadfile, env=env)
else:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight=net_arch
    policy_kwargs = dict(
        #activation_fn=torch.nn.ReLU,
        #net_arch=[dict(pi=[32, 32], vf=[32, 32])]
        #net_arch=[32, 32]
    )
    model = DQN('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)#, learning_rate=0.1)

try:
    #model.learn(total_timesteps=int(2e9))
    while True:
        # Maybe this isn't working? I think its fine?r
        model.learn(total_timesteps=int(2e5))
        if args.savefile:
            model.save(args.savefile)
            print('Saved Model')
except KeyboardInterrupt:
    pass

if args.savefile:
    model.save(args.savefile)

test(model, env)