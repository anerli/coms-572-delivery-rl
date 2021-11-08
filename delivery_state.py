import numpy as np
import random
from delivery_action import DeliveryAction


class Colors:
    # Blue-ish BG, black FG
    PLAYER = '\u001b[48;2;40;180;255;38;2;0;0;0m'
    # Yellowish BG, black FG
    PICKUP = '\u001b[48;2;200;200;0;38;2;0;0;0m'
    # Dark green BG, black FG
    DROPOFF = '\u001b[48;2;0;200;0;38;2;0;0;0m'

    RESET = '\u001b[0m'

class DeliveryState:
    '''
    x_lim: width of board.
    y_lim: height of board.
    dtype: data type of underlying np.arrays.
    step_lim: steps before marking as terminal.
    '''
    def __init__(self, x_lim, y_lim, dtype, step_lim=60):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.dtype = dtype
        self.step_lim = step_lim
        self.reset()

    def reset(self):
        # Step
        self.t = 0
        # Player coordinates
        self.player = np.array([2, 0], dtype=self.dtype)
        # Package Pickup Locations
        self.pickups = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)
        # Current Package Locations, value indicates num packages.
        self.packages = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)
        # Package Dropoff Locations
        self.dropoffs = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)

        # Create some pickup / dropoff locations by setting some of those array values to 1.
        # Sample without replacement:
        # space = [(x, y) for x in range(self.x_lim) for y in range(self.y_lim)]
        # pickup_loc = random.choice(space)
        # space.remove(pickup_loc)
        # self.pickups
        # print(space)

        self.pickups[4,2] = 1
        self.dropoffs[1, 3] = 1
        self.dropoffs[0, 0] = 1

    def to_array(self):
        # Need to convert self to tuple of np.arrays (dtype np.int8) and python ints
        # That way the returned array can be part of an observation space defined
        # based on what is returned here.
        return (self.player, self.pickups, self.packages, self.dropoffs)

    def render(self):
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
                    #print(' | ', end='')
                    print(end=' ')
            print()
            # if y < self.y_lim - 1:
            #     print('\n' + '-'*self.x_lim*3)
            # else:
            #     print()

    @classmethod
    def direction_to_vec(cls, direction_str):
        if direction_str == 'UP':
            return (0, -1)
        if direction_str == 'RIGHT':
            return (1, 0)
        if direction_str == 'DOWN':
            return (0, 1)
        if direction_str == 'LEFT':
            return (-1, 0)

    # Convert direction relative to the player to coordinates
    def direction_to_pos(self, direction_str):
        vec = DeliveryState.direction_to_vec(direction_str)
        return (self.player[0]+vec[0], self.player[1]+vec[1])

    def in_bounds(self, pos):
        return (0 <= pos[0] < self.x_lim) and (0 <= pos[1] < self.y_lim)

    def move(self, pos):
        if not self.in_bounds(pos):
            return
        self.player[0] = pos[0]
        self.player[1] = pos[1]

    def step(self, action):
        action_name = DeliveryAction(action).name
        arr = action_name.split('_')
        act = arr[0]
        direction = arr[1]
        pos = self.direction_to_pos(direction)
        if act == 'MOVE':
            self.move(pos)
        #print(act, direction)

        obs = self.to_array()
        reward = 0

        self.t += 1
        done = self.t >= self.step_lim
        info = {}
        return obs, reward, done, info