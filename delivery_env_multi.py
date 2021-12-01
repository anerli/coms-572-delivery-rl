import gym
from gym.spaces import Discrete, Box, Dict
import numpy as np
from delivery_state_multi import DeliveryState
from delivery_action_multi import SingleDeliveryAction

class DeliveryEnv(gym.Env):
    def __init__(self, init_state: DeliveryState):
        self.state = init_state

        x_lim = init_state.x_lim
        y_lim = init_state.y_lim
        dtype = init_state.dtype

        # https://gym.openai.com/docs/#spaces
        self.action_space = Discrete(len(SingleDeliveryAction)**self.state.num_players)

        player_high = np.zeros((self.state.num_players, 2))
        for i in self.state.num_players:
            player_high[i][0] = x_lim - 1
            player_high[i][1] = y_lim - 1
        
        self.observation_space = Dict({
            # Player coordinate
            'players': Box(
                #low=np.array([0, 0]),
                low=np.zeros((self.state.num_players, 2)),#np.array([0, 0])
                high=player_high,#np.array([x_lim-1, y_lim-1]),
                dtype=dtype
            ),
            # Pickup Locations
            'spawners': Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=dtype
            ),
            # Package Locations
            'packages': Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 9), # Say max of 9 packages at any given location (so we can display with one char)
                dtype=dtype
            ),
            # Dropoff Locations
            'dropoffs': Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=dtype
            ),
        })
        
    def reset(self):
        self.state.reset()
        return self.state.to_array()

    def step(self, action):
        obs, reward, done, info = self.state.step(action)
        return obs, reward, done, info
        
    def render(self, mode='human'):
        self.state.render()

def random_actions_test():
    init_state = DeliveryState.from_file('./envs/5x4/env.txt')
    env = DeliveryEnv(init_state)
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print('Action:', DeliveryAction(action).name)
        print('Reward:', reward)
        print('Resultant State:')
        env.render()
        print()

if __name__ == '__main__':
    random_actions_test()
    