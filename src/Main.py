from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch
from tqdm import tqdm

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

#def evaluate(env, num_episodes):
#    for _ in range(num_episodes):
        
env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording', force=True)
print('state_space / observation_space: ' + str(np.array(env.observation_space)))
print('action_space: ' + str(env.action_space))

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00005
total_rewards = []
agent = DQNAgent(env, 'saves/data.nn')
np.random.seed(1000)
#evaluating
for i in range(10):
    total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))
total_rewards = []
#training
print('\n\nTRAINING')
#agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
for i in tqdm (range (20), desc="Learning..."):
    agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1)
    #total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
    #print(total_reward)
    #total_rewards.append(total_reward)
    if i%1000 == 0:
        total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0)
        print(total_reward)
        agent.save('saves/data1.nn')
#agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
agent.save('saves/data1.nn')
#print(total_rewards)
#print('mean: ' + str(np.array(total_rewards).mean()))

print('\n\nEVALUATING')

total_rewards = []
#evaluating
for i in range(10):
    total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0, True)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))