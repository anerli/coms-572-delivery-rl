from stable_baselines3.common.evaluation import evaluate_policy
from delivery_action import DeliveryAction

def test(model, env):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    print(f'{mean_reward=}')

    # Render one episode
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        print('Action:', DeliveryAction(action).name)
        print('Reward:', reward)
        print('Resultant State:')
        env.render()
        print()