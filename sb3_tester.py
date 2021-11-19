from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from delivery_action import DeliveryAction
from delivery_state import DeliveryState
from delivery_env import DeliveryEnv

if __name__=='__main__':
    '''
    num_cpu = 4
    env_makers = []
    for i in range(num_cpu):
        init_state = DeliveryState(5, 5, 2, 1)
        env = DeliveryEnv(init_state)
        # For whatever reason it wants these as functons which return envs
        env_makers.append(lambda: env)
    env = SubprocVecEnv(env_makers)
    '''
    init_state = DeliveryState(5, 5, 2, 1)
    env = DeliveryEnv(init_state)

    model = PPO.load('ppo_delivery2', env=env)

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