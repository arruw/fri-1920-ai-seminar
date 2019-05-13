import gym
import numpy as np
import random
from IPython.display import clear_output
from solve import solve

env = gym.make("Taxi-v2").env
q_table = np.load(".tmp/taxi/q_table.npy")

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    
    epochs, penalties, reward, _, _ = solve(env, q_table)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")