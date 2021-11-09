# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
# https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
# https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3

# TODO: Look into vectorized environments for faster PPO training:
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html?highlight=vectorized
# Multiprocessing example: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/multiprocessing_rl.ipynb
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=multiprocessing#multiprocessing-unleashing-the-power-of-vectorized-environments

import gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv
from delivery_action import DeliveryAction

# Vector env gets mad if not in main method
if __name__ == '__main__':
    # Setup env
    init_state = DeliveryState(5, 5, 2, 1)
    env = DeliveryEnv(init_state)
    '''
    num_cpu = 4

    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=multiprocessing#multiprocessing-unleashing-the-power-of-vectorized-environments
    # Could also use make_vec_env() from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_util.py
    # Can't load this properly - try using make_vec_env()?
    
    env_makers = []
    for i in range(num_cpu):
        init_state = DeliveryState(5, 5, 2, 1)
        env = DeliveryEnv(init_state)
        # For whatever reason it wants these as functons which return envs
        env_makers.append(lambda: env)
    env = SubprocVecEnv(env_makers)
    '''

    #model = DQN('MultiInputPolicy', env, verbose=1)
    #model = DQN.load('dqn_delivery', env=env)
    model = PPO('MultiInputPolicy', env, verbose=1)
    #model = PPO.load('ppo_delivery2', env=env)

    try:
        while True:
            model.learn(total_timesteps=int(2e5))
            model.save('ppo_delivery2')
            print('Saved Model')
    except KeyboardInterrupt:
        pass

    model.save('ppo_delivery2')


    # try:
    #     model.learn(total_timesteps=int(2e9)) # 2 billion steps later... (just stop it when you want)
    # except KeyboardInterrupt:
    #     #model.save('ppo_delivery2')
    #     #model.save('dqn_delivery')
    #     pass
    # except EOFError:
    #     print('Caught ur EOF dumbass')
    #     pass
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

        print('Action:', DeliveryAction(action).name)
        print('Reward:', reward)
        print('Resultant State:')
        env.render()
        print()