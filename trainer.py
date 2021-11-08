import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from delivery_state import DeliveryState
from delivery_env import DeliveryEnv


# Setup env
init_state = DeliveryState(5, 4)
init_state.player = (2, 0)
init_state.spawners[4, 2] = 1
init_state.dropoffs[1, 3] = 1
init_state.dropoffs[0, 0] = 1
env = DeliveryEnv(init_state)

states = env.observation_space.shape
actions = env.action_space.n

print(f'{states=}')
print(f'{actions=}')