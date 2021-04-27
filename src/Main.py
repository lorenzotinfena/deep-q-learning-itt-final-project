from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch
from tqdm import tqdm

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording', force=True)
print('state_space / observation_space: ' + str(np.array(env.observation_space)))
print('action_space: ' + str(env.action_space))

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.000005
total_rewards = []
agent = DQNAgent(env, 'saves/data.weights')
np.random.seed(1000)
#evaluating
for i in range(10):
    total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 0, True)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))
total_rewards = []
#training
print('\n\nTRAINING')
#agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
#for i in tqdm (range (200), desc="Learning..."):
#    agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, False)
    #total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
    #print(total_reward)
    #total_rewards.append(total_reward)
    #if i%50 == 0:
    #    agent.save('saves/data.agent')
#agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
#agent.save('saves/data.nn')
#print(total_rewards)
#print('mean: ' + str(np.array(total_rewards).mean()))
print('\n\nEVALUATING')

total_rewards = []
#evaluating
for i in range(10):
    total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 0, True)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))