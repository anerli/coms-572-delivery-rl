import gym
from gym.spaces import Discrete, Box, Tuple
import numpy as np
from numpy import random
from delivery_state import DeliveryState
from delivery_action import DeliveryAction

class DeliveryEnv(gym.Env):
    '''
    x_lim:
    '''
    def __init__(self, init_state: DeliveryState):
        self.state = init_state

        # Debug
        print('DeliverEnv init_state:')
        self.state.render()
        print()
        # hmm doesn't seem to show up
        #with open('test.txt', 'w') as f:
        #    f.write('Im alive!')

        x_lim = init_state.x_lim
        y_lim = init_state.y_lim
        dtype = init_state.dtype

        # https://gym.openai.com/docs/#spaces
        self.action_space = Discrete(len(DeliveryAction))

        # Not all libaries support the Tuple space:
        # https://stackoverflow.com/questions/58964267/how-to-create-an-openai-gym-observation-space-with-multiple-features
        # RLlib claims to support Tuple spaces and Dict spaces:
        # https://docs.ray.io/en/latest/rllib-models.html#variable-length-complex-observation-spaces
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
        
    def reset(self):
        self.state.reset()
        return self.state.to_array()

    def step(self, action):
        obs, reward, done, info = self.state.step(action)
        return obs, reward, done, info
        
    def render(self, mode='human'):
        self.state.render()

def predefined_behavior_test():
    init_state = DeliveryState(5, 4)
    init_state.player = (2, 0)
    init_state.spawners[4, 2] = 1
    init_state.dropoffs[1, 3] = 1
    init_state.dropoffs[0, 0] = 1
    env = DeliveryEnv(init_state)

    test_actions = [
        # Move to pickup location
        DeliveryAction.MOVE_RIGHT,
        DeliveryAction.MOVE_RIGHT,
        DeliveryAction.MOVE_RIGHT,
        DeliveryAction.MOVE_DOWN,
        # Grab packages
        DeliveryAction.GRAB_DOWN,
        # Move to deposit
        DeliveryAction.MOVE_LEFT,
        DeliveryAction.MOVE_LEFT,
        DeliveryAction.MOVE_LEFT,
        DeliveryAction.MOVE_LEFT,
        # Deposit (should result in reward)
        DeliveryAction.DROP_UP,
    ]

    env.render()
    print()
    for action in test_actions:
        obs, reward, done, info = env.step(action)
        print('Action:', action.name)
        print('Reward:', reward)
        print('Resultant State:')
        env.render()
        print()

def random_actions_test():
    init_state = DeliveryState(5, 4)
    init_state.player = (2, 0)
    init_state.spawners[4, 2] = 1
    init_state.dropoffs[1, 3] = 1
    init_state.dropoffs[0, 0] = 1
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
    predefined_behavior_test()
    #random_actions_test()
    