import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random

# https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
class Colors:
    RED = '\u001b[31m'
    BLACK = '\u001b[30m'
    BG_RED = '\u001b[41m' # Use for player
    BG_BRIGHT_RED = '\u001b[41;1m'
    BG_MAGENTA = '\u001b[45m' # Use for pickup
    BG_GREEN = '\u001b[42m' # Use for dropoff
    BG_BLUE = '\u001b[44m'
    BG_YELLOW = '\u001b[43m'

    # Blue-ish BG
    PLAYER = '\u001b[48;2;40;180;255;38;2;0;0;0m'

    # Yellowish BG
    PICKUP = '\u001b[48;2;200;200;0;38;2;0;0;0m'

    # Dark green BG
    DROPOFF = '\u001b[48;2;0;200;0;38;2;0;0;0m'

    RESET = '\u001b[0m'

# Discrete data type used for np.array representations.
DTYPE = np.uint8

class DeliveryState:
    def __init__(self, x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.reset()

    def reset(self):
        global DTYPE
        # Player coordinates
        self.player = np.array([1, 0], dtype=DTYPE)
        # Package Pickup Locations
        self.pickups = np.zeros(shape=(self.x_lim, self.y_lim), dtype=DTYPE)
        # Current Package Locations, value indicates num packages.
        self.packages = np.zeros(shape=(self.x_lim, self.y_lim), dtype=DTYPE)
        # Package Dropoff Locations
        self.dropoffs = np.zeros(shape=(self.x_lim, self.y_lim), dtype=DTYPE)

        # TODO: Create some pickup / dropoff locations by setting some of those array values to 1.
        # Sample without replacement
        # For random:

        # space = [(x, y) for x in range(self.x_lim) for y in range(self.y_lim)]
        # pickup_loc = random.choice(space)
        # space.remove(pickup_loc)
        # self.pickups
        # print(space)

        self.pickups[2,2] = 1
        self.dropoffs[0, 2] = 1
        self.dropoffs[0, 1] = 1



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


        for y in range(self.y_lim):
            for x in range(self.x_lim):
                ch = ''
                if self.player[0] == x and self.player[1] == y:
                    # Always end in m, semi colon separated code vals
                    # 0m: clear
                    # 38: set color (next args define color)
                    # 48: set bg color
                    ch += Colors.PLAYER#Colors.BG_BRIGHT_RED + '\u001b[38;5;232m' #Colors.BLACK
                elif self.pickups[x, y] > 0:
                    ch += Colors.PICKUP#Colors.BG_MAGENTA
                elif self.dropoffs[x, y] > 0:
                    ch += Colors.DROPOFF#Colors.BG_BLUE

                ch += str(self.packages[x, y])
                
                ch += Colors.RESET

                print(ch, end='')
                if x < self.x_lim - 1:
                    print(end=' ')
            print()

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
                high=np.full((x_lim, y_lim), 9), # Say max of 9 packages at any given location (so we can display with one char)
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
        

    def reset(self):
        self.state.reset()
        return self.state.to_array()

    def step(self, action):
        pass

    def render(self, mode='human'):
        self.state.render()
    
if __name__ == '__main__':
    env = DeliveryEnv(3, 4)
    # print('Action Space Shape:', env.action_space.n)
    # print('Action Space Samples:')
    # for _ in range(10):
    #     print(env.action_space.sample())
    # print('Observation Space Shape:', env.observation_space.shape)
    # print('Observation Space Samples:')
    # for _ in range(10):
    #     print(env.observation_space.sample())

    env.render()