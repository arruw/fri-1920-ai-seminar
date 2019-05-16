import gym
import numpy as np
from DQNAgent import DQNAgent


# initialize gym environment and the agent
env = gym.make('CartPole-v1')
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load('.tmp/cartpole-dqn/model.h5')
agent._epsilon = 0.00

for _ in range(500):

    total_reward = 0

    # reset state in the beginning of each game
    state = env.reset()
    state = np.reshape(state, [1, 4])

    while True:
        # turn this on if you want to render
        env.render()
        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        # make next_state the new current state for the next frame.
        state = next_state
        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            print("Done. Reward: ", total_reward)
            break

    # agent.replay(32)

env.close()