import gym
import numpy as np
from DQNAgent import DQNAgent
import pandas as pd

df = pd.DataFrame(columns=['example','environment','episode','reward','epsilon'])

# initialize gym environment and the agent
env = gym.make('CartPole-v1')
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
episodes = 500000
master_level = 0
# Iterate the game
for e in range(episodes):
    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 4])
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    for time_t in range(500):
        # turn this on if you want to render
        # env.render()
        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, score: {}".format(e, episodes, time_t))
            df.loc[e] = ['cartpole-dqn','CartPole-v1',e,time_t,agent._epsilon]

            if time_t >= 400:
                master_level+=1
            else:
                master_level = 0

            break
    # train the agent with the experience of the episode
    agent.replay(32)

    if master_level > 25:
        break

env.close()
agent.save('.tmp/cartpole-dqn/model.h5')

df.to_csv ('.tmp/cartpole-dqn/data.csv', index = None, header=True)