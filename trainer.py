from stable_baselines3 import DQN, PPO

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

from tester import test

import torch

import os
from os.path import join

from argparse import ArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ENVS_DIR = join(SCRIPT_DIR, 'envs')

print(SCRIPT_DIR)

parser = ArgumentParser()
#parser.add_argument('-l', '--loaddir', type=str, required=False)
#parser.add_argument('-s', '--savedir', type=str, required=False)
parser.add_argument('-d', '--dir', type=str, required=True)
parser.add_argument('-m', '--modelname', type=str, required=False, default='model')
parser.add_argument('-e', '--steplimit', type=int, default=20, required=False)
parser.add_argument('-a', '--algorithm', type=str, required=False, default='dqn')
parser.add_argument('-t', '--test', action='store_true')
args = parser.parse_args()

ENV_DIR = join(ENVS_DIR, args.dir)

if not os.path.isdir(ENV_DIR):
    #raise Exception('Directory  does not exist.')
    print(f'Directory {ENV_DIR} does not exist, creating it and adding a blank env.txt to be filled out.')
    os.mkdir(ENV_DIR)
    with open(join(ENV_DIR, 'env.txt'), 'w') as f:
        pass
    exit()

#init_state = DeliveryState(5, 5, (0,0), [(4, 3), (0, 4)], [(4, 0)])
init_state = DeliveryState.from_file(join(ENV_DIR, 'env.txt'))
if args.steplimit:
    init_state.step_lim = args.steplimit
env = DeliveryEnv(init_state)

MODEL_PATH = join(ENV_DIR, args.modelname)

print(f'{MODEL_PATH=}')

if args.algorithm.lower() == 'dqn':
    ModelClass = DQN
elif args.algorithm.lower() == 'ppo':
    ModelClass = PPO
else:
    raise Exception('Invalid algorithm.')

if os.path.isfile(MODEL_PATH + '.zip'):
    print('Model exists, loading it...')
    model = ModelClass.load(MODEL_PATH, env=env)

    #model = DQN.load('envs/5x4/model')
else:
    print('Model does not exist, creating it...')
    model = ModelClass('MlpPolicy', env, verbose=1)#, learning_rate=0.01)

# if args.loadfile:
#     model = DQN.load(args.loadfile, env=env)
# else:
#     # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html?highlight=net_arch
#     policy_kwargs = dict(
#         #activation_fn=torch.nn.ReLU,
#         #net_arch=[dict(pi=[32, 32], vf=[32, 32])]
#         #net_arch=[32, 32]
#     )
#     model = DQN('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs)#, learning_rate=0.1)

if not args.test:
    try:
        #model.learn(total_timesteps=int(2e9))
        while True:
            # Maybe this isn't working? I think its fine?r
            model.learn(total_timesteps=int(2e5))

            model.save(MODEL_PATH)
            # if args.savefile:
            #     model.save(args.savefile)
            #     print('Saved Model')
    except KeyboardInterrupt:
        pass
    model.save(MODEL_PATH)

# if args.savefile:
#     model.save(args.savefile)

test(model, env)