from operator import xor
import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random

class Colors:
    RED = '\u001b[31m'
    BG_RED = '\u001b[41m' # Use for player

    BG_MAGENTA = '\u001b[45m' # Use for pickup

    BG_GREEN = '\u001b[42m' # Use for dropoff

    RESET = '\u001b[0m'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Discrete data type used for np.array representations.
DTYPE = np.uint8

class DeliveryState:
    def __init__(self, x_lim, y_lim):
        self.reset()

    def reset(self):
        global DTYPE
        # Player coordinates
        self.player = np.array([0, 0], dtype=DTYPE)
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
        self.observation_space = Tuple((
            # Player coordinate
            Box(
                low=np.array([0, 0]),
                high=np.array([x_lim-1, y_lim-1]),
                dtype=DTYPE
            ),
            # Pickup Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=DTYPE
            ),
            # Package Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 10), # Say max of 10 packages at any given location
                dtype=DTYPE
            ),
            # Dropoff Locations
            Box(
                low=np.zeros((x_lim, y_lim)),
                high=np.full((x_lim, y_lim), 1),
                dtype=DTYPE
            ),
        ))

        self.state = DeliveryState(x_lim, y_lim)
        # self.observation_space = gym.spaces.Tuple((gym.spaces.Box(
        #     low=np.array([0,0], dtype=dtype),
        #     high=np.array([100, 20000], dtype=dtype),
        #     dtype=dtype
        # ), gym.spaces.Discrete(4)))
        #self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))

        #self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))

        #_ = self.reset()
        

    def reset(self):
        self.state.reset()
        return self.state.to_array()

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass
    
if __name__ == '__main__':
    env = DeliveryEnv(3, 3)
    print('Action Space Shape:', env.action_space.n)
    print('Action Space Samples:')
    for _ in range(10):
        print(env.action_space.sample())
    print('Observation Space Shape:', env.observation_space.shape)
    print('Observation Space Samples:')
    for _ in range(10):
        print(env.observation_space.sample())