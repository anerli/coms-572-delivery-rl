import numpy as np
import random
from delivery_action_multi import SingleDeliveryAction, get_actions
from colors import Colors
from utils import manhattan_dist
import math
from typing import List

class DeliveryState:
    '''
    x_lim: width of board.
    y_lim: height of board.
    dtype: data type of underlying np.arrays.
    step_lim: steps before marking as terminal.
    '''
    def __init__(self, x_lim, y_lim, initial_player_positions: List[tuple], 
        spawner_positions: List[tuple], dropoff_positions: List[tuple],
        step_lim=60, dtype=np.uint8):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.dtype = dtype
        self.step_lim = step_lim

        self.carry_lim = 3
        # Max packages at any position
        self.package_max = 9

        #self.init_player_pos = init_player_pos
        self.initial_player_positions = initial_player_positions

        # Package Pickup Locations
        self.spawners = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)
        # Package Dropoff Locations
        self.dropoffs = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)

        for pos in spawner_positions:
            self.spawners[pos] = 1
        for pos in dropoff_positions:
            self.dropoffs[pos] = 1

        self.debug = False
        
        # == Reward params ==
        # Reward for each package delivered
        self.reward_delivery = 1000
        # Multiplier for reward gained by minimizing the distance between packages and their dropoffs.
        self.reward_package_dest_dist_multiplier = 2.0
        # Multiplier for reward gained by minimizing distance between self and packages.
        self.reward_self_package_dist_multiplier = 1.0
        # Reward per step for simply holding a package
        self.reward_package_hold = 0
        # Penalty for being an idiot, like trying to go outside bounds or running into something
        # (penalty for making an invalid move, which results in a pass)
        self.idiot_penalty = 0 # Consider increasing

        self.reset()

    @classmethod
    def from_file(cls, fpath):
        dropoff_ch = 'd'
        spawner_ch = 's'
        player_ch = 'p'
        with open(fpath, 'r') as f:
            s = f.read()
        lines = s.splitlines()

        x_lim = None
        y_lim = len(lines) - 1

        #player_pos = None
        player_positions = []
        spawner_positions = []
        dropoff_positions = []

        step_lim = int(lines[0])

        for y, line in enumerate(lines[1:]):
            # Remove ALL whitespace
            # Each symbol should be one character
            line = ''.join(line.split())
            if x_lim is None:
                x_lim = len(line)
            else:
                if len(line) != x_lim:
                    raise Exception('All rows must be the same length.')
            
            for x, ch in enumerate(line):
                if ch == player_ch:
                    player_positions.append((x, y))
                    # if player_pos is not None:
                    #     raise Exception('Multiple players not yet supported.')
                    # player_pos = (x, y)
                elif ch == spawner_ch:
                    spawner_positions.append((x, y))
                elif ch == dropoff_ch:
                    dropoff_positions.append((x, y))
        
        return cls(x_lim, y_lim, player_positions, spawner_positions, dropoff_positions, step_lim)

    @property
    def num_players(self):
        return len(self.initial_player_positions)
            

    def reset(self):
        # Step
        self.t = 0

        # For logging
        self.packages_delivered = 0
        
        # Reset Player position
        #self.player = self.init_player_pos
        self.players = np.zeros(shape=(self.num_players, 2), dtype=self.dtype)
        for i, pos in enumerate(self.initial_player_positions):
            self.players[i][0] = pos[0]
            self.players[i][1] = pos[1]
        #self.player_dir = 'UP'
        # Current Package Locations, value indicates num packages.
        self.packages = np.zeros(shape=(self.x_lim, self.y_lim), dtype=self.dtype)

    def to_array(self):
        # Need to convert self to tuple of np.arrays (dtype np.int8) and python ints
        # That way the returned array can be part of an observation space defined
        # based on what is returned here.
        return {'players': self.players, 'spawners': self.spawners, 'packages': self.packages, 'dropoffs': self.dropoffs}

    def render(self):
        for y in range(self.y_lim):
            for x in range(self.x_lim):
                ch = ''
                #if self.player[0] == x and self.player[1] == y:
                is_player = False
                for i in range(self.num_players):
                    if tuple(self.players[i]) == (x, y):
                        is_player = True
                if is_player:
                    ch += Colors.PLAYER
                elif self.spawners[x, y] > 0:
                    ch += Colors.PICKUP
                elif self.dropoffs[x, y] > 0:
                    ch += Colors.DROPOFF

                ch += str(self.packages[x, y])
                
                ch += Colors.RESET

                print(ch, end='')
                if x < self.x_lim - 1:
                    print(end=' ')
            print()


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
    def direction_to_pos(self, player_idx,  direction_str):
        vec = DeliveryState.direction_to_vec(direction_str)
        return (tuple(self.players[player_idx])[0]+vec[0], tuple(self.players[player_idx])[1]+vec[1])

    def in_bounds(self, pos):
        return (0 <= pos[0] < self.x_lim) and (0 <= pos[1] < self.y_lim)

    def occupied(self, pos):
        # Apparently can pass a tuple to np indexer
        # equivalent to [pos[0], pos[1]]

        for i in range(self.num_players):
            if tuple(self.players[i]) == pos:
                return True
        # Player blocked by drop offs, spawners, and any stray packages
        return self.dropoffs[pos] > 0 or self.spawners[pos] > 0 or self.packages[pos] > 0

    def move_packages(self, from_pos, to_pos, lim=None):
        if lim is not None:
            lim = max(0, lim)
            transfer_amt = min(lim, self.packages[from_pos])
            self.packages[to_pos] += transfer_amt
            self.packages[from_pos] -= transfer_amt
        else:
            self.packages[to_pos] = self.packages[from_pos]
            self.packages[from_pos] = 0

    def move(self, player_idx,  pos):
        if self.occupied(pos):
            return self.idiot_penalty

        # ====== Calculate package move reward ======
        # Calculate dist to closest dropoff before and after
        reward = 0
        if self.packages[tuple(self.players[player_idx])] > 0:
            closest_dropoff_pos_before = None
            closest_dist_before = math.inf
            closest_dropoff_pos_after = None
            closest_dist_after = math.inf
            for x in range(self.x_lim):
                for y in range(self.y_lim):
                    if self.dropoffs[x, y] > 0:
                        dist_before = manhattan_dist(tuple(self.players[player_idx]), (x,y))
                        if dist_before < closest_dist_before:
                            closest_dropoff_pos_before = (x, y)
                            closest_dist_before = dist_before
                        dist_after = manhattan_dist(pos, (x,y))
                        if dist_after < closest_dist_after:
                            closest_dropoff_pos_after = (x, y)
                            closest_dist_after = dist_after

            if self.debug:
                print(f'{closest_dist_before=}')
                print(f'{closest_dropoff_pos_before=}')
                print(f'{closest_dist_after=}')
                print(f'{closest_dropoff_pos_after=}')
            improvement = closest_dist_before - closest_dist_after
            reward_change = improvement * self.packages[tuple(self.players[player_idx])] * self.reward_package_dest_dist_multiplier
            reward += reward_change 
        else:
            # No packages, incentivise move toward packages
            #                     Just some finite unreachably large distance, math.inf causes issues when no packages on board
            closest_dist_before = self.x_lim * self.y_lim
            closest_dist_after = self.x_lim * self.y_lim 
            for x in range(self.x_lim):
                for y in range(self.y_lim):
                    if self.packages[x, y] > 0:
                        dist_before = manhattan_dist(tuple(self.players[player_idx]), (x,y))
                        if dist_before < closest_dist_before:
                            #closest_package_pos_before = (x, y)
                            closest_dist_before = dist_before
                        dist_after = manhattan_dist(pos, (x,y))
                        if dist_after < closest_dist_after:
                            #closest_package_pos_after = (x, y)
                            closest_dist_after = dist_after
            improvement = closest_dist_before - closest_dist_after
            reward_change = improvement * self.reward_self_package_dist_multiplier
            reward += reward_change
        # ============================================

        # Move the packages the player is carrying to where the player will be.
        self.move_packages(tuple(self.players[player_idx]), pos, self.carry_lim)

        # Move the player
        #tuple(self.players[player_idx]) = pos
        self.players[player_idx][0] = pos[0]
        self.players[player_idx][1] = pos[1]

        return reward

    def grab(self, player_idx,  pos):
        if self.packages[pos] == 0:
            return self.idiot_penalty
        
        lim = self.carry_lim - self.packages[tuple(self.players[player_idx])]
        #lim = min(lim, self.package_max - self.self.packages[pos])
        self.move_packages(pos, tuple(self.players[player_idx]), lim)
        return 0

    '''
    Do drop action and return any reward gained from depositing at a dropoff.
    '''
    def drop(self, player_idx,  pos):
        if self.packages[tuple(self.players[player_idx])] == 0:
            return self.idiot_penalty

        packages_deposited = 0
        # Don't need to use carry limit here probably? Let's say players can drop more than they can carry
        #lim = self.carry_lim - self.packages[tuple(self.players[player_idx])]
        # Check if there is a player at pos and if so make sure recieving player isn't carrying too much
        is_player = False
        for i in range(self.num_players):
            if tuple(self.players[i]) == pos:
                is_player = True
                break
        if is_player:
            recieving_player_packages = self.packages[pos]
            lim = self.carry_lim - recieving_player_packages#min(self.carry_lim, recieving_player_packages)
        else:
            lim = self.carry_lim
        lim = min(lim, self.package_max - self.packages[pos])
        self.move_packages(tuple(self.players[player_idx]), pos, lim)
        if self.dropoffs[pos] > 0:
            packages_deposited += self.packages[pos]
            # Clear packages to "deposit" them.
            self.packages[pos] = 0

        self.packages_delivered += packages_deposited
        #print(self.packages_delivered)
        # Reward
        return packages_deposited * self.reward_delivery
    
    def step_player(self, player_idx, player_action):
        reward = 0.0

        # === Handle Agent Action ===
        action_name = SingleDeliveryAction(player_action).name

        # TODO: Fix below for multi
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        if action_name == 'GRAB':
            # Grab at pos with most packages
            best_pos = None
            most = 0
            for direction in directions:
                pos = self.direction_to_pos(player_idx, direction)
                if self.in_bounds(pos) and (best_pos is None or self.packages[pos] >= most):
                        most = self.packages[pos]
                        best_pos = pos
            if not self.in_bounds(best_pos):
                # In case somehow we are surrounded by invalid positions
                reward += self.idiot_penalty
            else:
                reward += self.grab(player_idx, best_pos)
        elif action_name == 'DROP':
            # Drop at dropoff or at available location. o.w. drop to the left
            best_pos = None
            for direction in directions:
                pos = self.direction_to_pos(player_idx, direction)
                if self.in_bounds(pos) and (best_pos is None or self.dropoffs[pos] > 0):
                    best_pos = pos
            if not self.in_bounds(best_pos):
                # In case somehow we are surrounded by invalid positions
                drop_reward = self.idiot_penalty
            else:
                drop_reward = self.drop(player_idx, best_pos)
            if self.debug:
                print('Reward due to drop:', drop_reward)
            reward += drop_reward
        else:
            pos = self.direction_to_pos(player_idx, action_name)
            if not self.in_bounds(pos):
                reward += self.idiot_penalty
            else:
                movement_reward = self.move(player_idx, pos)
                if self.debug:
                    print('Reward due to movement:', movement_reward)
                reward += movement_reward

        # === Misc Rewards ===
        package_hold_reward = self.reward_package_hold * self.packages[tuple(self.players[player_idx])]
        if self.debug: print('Reward due to holding packages:', package_hold_reward)
        reward += package_hold_reward

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

        return reward


    def step(self, action):
        reward = 0.0

        player_actions = get_actions(action, self.num_players)

        for player_idx in range(self.num_players):
            reward += self.step_player(player_idx, player_actions[player_idx])


        # === Return ===
        obs = self.to_array()

        self.t += 1
        done = self.t >= self.step_lim
        info = {}

        return obs, reward, done, info