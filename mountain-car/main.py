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
    best_reward_list = []
    
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
            else:
                # lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action]
                delta = learning*(reward+discount*np.max(Q[state2[0],state2[1]]) - 
                                 Q[state1[0], state1[1],action])
                Q[state1[0], state1[1],action] += delta
                                     
            # Update total_reward and current state
            total_reward += reward
            state1 = state2
        
        # Reduce epsilon over time
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(total_reward)
        
        if (i+1) % 100 == 0:
            avg_reward = np.mean(reward_list)
            max_reward = np.max(reward_list)
            avg_reward_list.append(avg_reward)
            best_reward_list.append(max_reward)
            print("Episode ", i+1, "\nAverage Reward: ", avg_reward, "\nBest reward: ", max_reward,"\n-----------")
            reward_list = []
            
    env.close()
    
    return avg_reward_list, best_reward_list

# Run Q-learning algorithm
# Epsilon - how much we want to explore rate
# env, learning rate, discount rate, epsilon, min_eps, episodes):
# TODO: Loop through different parameters, plot results
# Learning rate 0.05, 0.1, 0.2, etc.
learning_rates = [0.05, 0.1, 0.2]
discount_rates = [0.8, 0.9]
exploration_rates = [0.2, 0.8]

for learning_rate in learning_rates:
    for discount_rate in discount_rates:
        for exploration_rate in exploration_rates:
            rewards, best_rewards = QLearning(env, learning_rate, discount_rate, exploration_rate, 0, 5000)
            
            # TODO: Plot differences when applying different learning rate, discount rate etc.
            # TODO: Different Epsilon values yield different results....debate EXPLOITATION vs EXPLORATION
            # Plot average rewards over time
            plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
            plt.xlabel('Episode')
            plt.ylabel('AVG Reward')
            plt.title('Average reward with episodes')
            plt.savefig('./plots/lr_'+str(learning_rate)+'_dr_'+str(discount_rate)
                +'_er_'+str(exploration_rate)+'.jpg')     
            plt.close() 

            # Plot best rewards over time
            # plt.plot(100*(np.arange(len(best_rewards)) + 1), best_rewards)
            # plt.xlabel('Episode')
            # plt.ylabel('Best Reward')
            # plt.title('Best reward with episodes')
            # plt.savefig('./plots/best_rewards_'+str(learning_rate)+'.jpg')     
            # plt.close() 
