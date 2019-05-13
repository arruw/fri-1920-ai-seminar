import gym
import random
import numpy as np
from solve import solve
from IPython.display import clear_output

env = gym.make("Taxi-v2").env
q_table = np.zeros([env.observation_space.n, env.action_space.n])

"""Training the agent"""

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, q_table, _ = solve(env, q_table)

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

"""Training finished."""

np.save(".tmp/taxi/q_table.npy", q_table)