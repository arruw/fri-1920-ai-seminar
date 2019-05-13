import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep
from solve import solve

env = gym.make("Taxi-v2").env
q_table = np.load(".tmp/taxi/q_table.npy")

# env.s = 328  # set environment to illustration's state

epochs, penalties, _, _, frames = solve(env, q_table)
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)