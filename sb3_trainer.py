# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
# https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
# https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3

# TODO: Look into vectorized environments for faster PPO training:
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html?highlight=vectorized

import gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

# Setup env
init_state = DeliveryState(5, 4, (2, 0))
init_state.spawners[4, 2] = 1
init_state.dropoffs[1, 3] = 1
init_state.dropoffs[0, 0] = 1

env = DeliveryEnv(init_state)

#model = DQN('MultiInputPolicy', env, verbose=1)
#model = DQN.load('dqn_delivery', env=env)
#model = PPO('MultiInputPolicy', env, verbose=1)
model = PPO.load('ppo_delivery', env=env)

try:
    model.learn(total_timesteps=int(2e9)) # 2 billion steps later... (just stop it when you want)
except KeyboardInterrupt:
    model.save('ppo_delivery')
    #model.save('dqn_delivery')

'''
To load:
model = DQN.load('dqn_delivery', env=env)
'''

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'{mean_reward=}')

# Enjoy trained agent
obs = env.reset()
for i in range(60):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    print('Action:', action.name)
    print('Reward:', reward)
    print('Resultant State:')
    env.render()
    print()