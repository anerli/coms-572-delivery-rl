import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from delivery_state import DeliveryState
from delivery_action import DeliveryAction


# Discrete data type used for np.array representations.
#DTYPE = np.uint8

# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences


class DeliveryEnv(gym.Env):
    '''
    x_lim:
    '''
    def __init__(self, x_lim, y_lim, dtype=np.uint8):
        #global DTYPE
        # https://gym.openai.com/docs/#spaces
        # Can use spaces.Tuple to compose spaces
        '''
        Actions:
        - Move Up/Right/Down/Left (4)
        - Grab Up/Right/Down/Left (4)
        - Drop Up/Right/Down/Left (4)
        '''
        self.action_space = Discrete(len(DeliveryAction))

        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        #dtype = np.uint8
        self.observation_space = Tuple((
            # Player coordinate
            Box(
                low=np.array([0, 0]),
                high=np.array([x_lim-1, y_lim-1]),
                dtype=dtype
            ),
            # Pickup Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=dtype
            ),
            # Package Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 9), # Say max of 9 packages at any given location (so we can display with one char)
                dtype=dtype
            ),
            # Dropoff Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=dtype
            ),
        ))

        self.state = DeliveryState(x_lim, y_lim, dtype=dtype)
        

    def reset(self):
        self.state.reset()
        return self.state.to_array()

    def step(self, action):
        return self.state.step(action)
        
    def render(self, mode='human'):
        self.state.render()
        print()
    
if __name__ == '__main__':
    env = DeliveryEnv(5, 4)
    # print('Action Space Shape:', env.action_space.n)
    # print('Action Space Samples:')
    # for _ in range(10):
    #     print(env.action_space.sample())
    # print('Observation Space Shape:', env.observation_space.shape)
    # print('Observation Space Samples:')
    # for _ in range(10):
    #     print(env.observation_space.sample())

    env.render()

    env.step(DeliveryAction.MOVE_RIGHT)

    env.render()

    env.step(DeliveryAction.MOVE_RIGHT)

    env.render()
    
    env.step(DeliveryAction.MOVE_RIGHT)

    env.render()