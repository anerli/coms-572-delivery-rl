import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random

DTYPE = np.uint8

class DeliveryState:
    def __init__(self, x_lim, y_lim):
        global DTYPE
        # Player coordinates
        self.player = np.array([0, 0], dtype=DTYPE)

        '''
        Notice:
        These arrays can be accessed by [x, y] but they are stored
        as though each subarray is a column.

        So 
        [
            [a, b, c],
            [1, 2, 3]
        ]

        Represents the game state:
        a 1
        b 2
        c 3

        -> This is why we have the render function to make it prettier anyway.

        This confusion is necessary due to conflicting coventions of matrix
        indexing and digital 2D space representation. (i.e. (row, col)~(y, x) vs (x, y))
        '''

        # Package Pickup Locations
        self.pickups = np.zeros(shape=(x_lim, y_lim), dtype=DTYPE)
        # Current Package Locations, value indicates num packages.
        self.packages = np.zeros(shape=(x_lim, y_lim), dtype=DTYPE)
        # Package Dropoff Locations
        self.dropoffs = np.zeros(shape=(x_lim, y_lim), dtype=DTYPE)

    def to_array(self):
        # Need to convert self to tuple of np.arrays (dtype np.int8) and python ints
        # That way the returned array can be part of an observation space defined
        # based on what is returned here.
        return (self.player, self.pickups, self.packages, self.dropoffs)

    def render(self):
        print(f'{self.player=}')
        print(f'{self.pickups=}')
        print(f'{self.packages=}')
        print(f'{self.dropoffs=}')

class DeliveryEnv(gym.Env):
    '''
    x_lim:
    '''
    def __init__(self, x_lim, y_lim):
        global DTYPE
        # https://gym.openai.com/docs/#spaces
        # Can use spaces.Tuple to compose spaces
        '''
        Actions:
        - Move Up/Right/Down/Left (4)
        - Grab Up/Right/Down/Left (4)
        - Drop Up/Right/Down/Left (4)
        '''
        self.action_space = Discrete(12)

        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        #dtype = np.uint8
        self.overvation_space = Tuple((
            Box()
        ))
        # self.observation_space = gym.spaces.Tuple((gym.spaces.Box(
        #     low=np.array([0,0], dtype=dtype),
        #     high=np.array([100, 20000], dtype=dtype),
        #     dtype=dtype
        # ), gym.spaces.Discrete(4)))
        #self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))

        #self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))

        #_ = self.reset()
        

    def reset(self):
        pass
        #return np.array([0, 0])

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass
    
if __name__ == '__main__':
    env = DeliveryEnv()
    print('Action Space Shape:', env.action_space.n)
    print('Action Space Samples:')
    for _ in range(10):
        print(env.action_space.sample())
    print('Observation Space Shape:', env.observation_space.shape)
    print('Observation Space Samples:')
    for _ in range(10):
        print(env.observation_space.sample())