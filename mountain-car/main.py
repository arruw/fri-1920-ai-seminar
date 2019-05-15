import gym
import matplotlib.pyplot as plt
import numpy as np
import math

env = gym.make('MountainCar-v0')
# Set initial state of environment
env.reset()

# print("State space:", env.observation_space)
# print("Action space:", env.action_space)

# State space first value -1.2 to 0.6 (Position) -> *10
# State space second value  -0.07 to 0.07 (Velocity) -> *100
# We will need to discretize the state space, multiply x10, x100

# Q learning
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
    num_states = [math.ceil(x) for x in num_states]
    
    # Initialize Q table with zeroes
    # Q = np.zeros([num_states[0], num_states[1], env.action_space.n])
    # Initialize Q table with random values -1 to 1, or 0 to 1 etc. Test
    Q = np.random.uniform(low=0, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    # Track rewards
    reward_list = []
    avg_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        total_reward = 0
        reward = 0
        state = env.reset()
        # Discretize starting state
        state1 = (state - env.observation_space.low)*np.array([10, 100])
        state1 = np.round(state1, 0).astype(int)

        while done != True:   
            # Render environment for last five episodes - visualization
            # if i >= (episodes - 20):
            #     env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state1[0], state1[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward from action
            state_new, reward, done, info = env.step(action) 
            
            # Discretize state2, add the minimum, so we dont overflow
            state2 = (state_new - env.observation_space.low)*np.array([10, 100])
            state2 = np.round(state2, 0).astype(int)
            
            # Check if goal is reached
            if done and state_new[0] >= 0.5:
                Q[state1[0], state1[1], action] = reward
                
            # Adjust Q value for current state
            # new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2[0], 
                                                   state2[1]]) - 
                                 Q[state1[0], state1[1],action])
                Q[state1[0], state1[1],action] += delta
                                     
            # Update total_reward and current state
            total_reward += reward
            state1 = state2
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(total_reward)
        
        if (i+1) % 100 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            print("Episode ", i+1, "\nAverage Reward: ", avg_reward, "\nBest reward: ", np.max(reward_list),"\n-----------")
            reward_list = []
            
    env.close()
    
    return avg_reward_list

# Run Q-learning algorithm
# env, learning rate, discount rate, epsilon, min_eps, episodes):
# TODO: Loop through different parameters, plot results
# Learning rate 0.05, 0.1, 0.2, etc.
rewards = QLearning(env, 0.1, 0.9, 0.8, 0, 5000)

# Plot average rewards over time
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episode')
plt.ylabel('AVG Reward')
plt.title('Average reward with episodes')
plt.savefig('rewards.jpg')     
plt.close()    

# TODO: Plot best reward over time
# TODO: Plot differences when applying different learning rate, discount rate etc.