from core.CustomNeuralNetwork import CustomNeuralNetwork
import gym
import numpy as np
import torch
from tqdm import tqdm
from CartPoleWrapper import CartPoleWrapper


LR = 0.001
gamma = 0.95
exploration_epsilon = 0.5                
def main():
    np.random.seed(1000)

    # Global variables
    NUM_EPISODES = 1000
    MAX_TIMESTEPS = 1000
    nn = CustomNeuralNetwork([4, 5, 5, 2])
    env=CartPoleWrapper(gym.make('CartPole-v1'))

    # The main program loop
    for i_episode in range(NUM_EPISODES):
        observation = env.reset()
        # Iterating through time steps within an episode
        for t in range(MAX_TIMESTEPS):
            z, a = nn.forward_propagate(observation)
            q_values_predicted = a[-1]
            action = env.action_space.sample()  if np.random.uniform(0, 1) < exploration_epsilon else np.argmax(q_values_predicted)
            prev_obs = observation
            observation, reward, done, info = env.step(action)

            z1, a1 = nn.forward_propagate(observation)
            next_action_values = a1[-1]
            experimental_values = np.copy(q_values_predicted)
            if done:
                experimental_values[action] = -1
            else:
                experimental_values[action] = 1 + gamma*np.max(next_action_values)
            
        
            nn.backpropagate(z, a, experimental_values, LR)
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                print('Episode {} ended after {} timesteps'.format(i_episode, t+1))
                break
if __name__ == '__main__':
    main()