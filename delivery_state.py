import numpy as np
import random
from delivery_action import DeliveryAction
from colors import Colors

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
        self.player = (0, 0)#np.array([0, 0], dtype=self.dtype)
        # Package Pickup Locations
        self.spawners = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)
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

        # self.spawners[4,2] = 1
        # self.dropoffs[1, 3] = 1
        # self.dropoffs[0, 0] = 1

    def to_array(self):
        # Need to convert self to tuple of np.arrays (dtype np.int8) and python ints
        # That way the returned array can be part of an observation space defined
        # based on what is returned here.
        return (np.array(self.player, dtype=self.dtype), self.spawners, self.packages, self.dropoffs)

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
                elif self.spawners[x, y] > 0:
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

    def occupied(self, pos):
        # Apparently can pass a tuple to np indexer
        # equivalent to [pos[0], pos[1]]
        # Player blocked by drop offs, spawners, and any stray packages
        return self.dropoffs[pos] > 0 or self.spawners[pos] > 0 or self.packages[pos] > 0

    def move_packages(self, from_pos, to_pos):
        self.packages[to_pos[0], to_pos[1]] = self.packages[from_pos[0], from_pos[1]]
        self.packages[from_pos[0], from_pos[1]] = 0

    def move(self, pos):
        if self.occupied(pos):
            return
        # Move the packages the player is carrying to where the player will be.
        self.move_packages(self.player, pos)

        # Move the player
        self.player = pos

    def grab(self, pos):
        self.move_packages(pos, self.player)

    '''
    Do drop action and return any reward gained from depositing at a dropoff.
    '''
    def drop(self, pos):
        packages_deposited = 0
        self.move_packages(self.player, pos)
        if self.dropoffs[pos[0], pos[1]] > 0:
            packages_deposited += self.packages[pos[0], pos[1]]
            # Clear packages to "deposit" them.
            self.packages[pos[0], pos[1]] = 0
        # Reward
        return packages_deposited
        

    def step(self, action):
        reward = 0

        # === Handle Agent Action ===
        action_name = DeliveryAction(action).name
        arr = action_name.split('_')
        act = arr[0]
        direction = arr[1]
        pos = self.direction_to_pos(direction)

        if not self.in_bounds(pos):
            pass
        elif act == 'MOVE':
            self.move(pos)
        elif act == 'GRAB':
            self.grab(pos)
        elif act == 'DROP':
            reward += self.drop(pos)
        #print(act, direction)


        # === Spawn Packages ===
        # This could go before or after player action,
        # but I think it makes sense for the agent to only grab packages
        # seen in the last state, not newly generated ones.
        package_spawn_chance = 0.5
        for x in range(self.x_lim):
            for y in range(self.y_lim):
                if self.spawners[x, y] > 0 and self.packages[x,y] < 9:
                    if random.random() < package_spawn_chance:
                        self.packages[x,y] += 1


        # === Return ===
        obs = self.to_array()

        self.t += 1
        done = self.t >= self.step_lim
        info = {}
        return obs, reward, done, info