from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording', force=True)
print('state_space / observation_space: ' + str(np.array(env.observation_space)))
print('action_space: ' + str(env.action_space))

DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.1
total_rewards = []
agent = DQNAgent(env)
#evaluating
for i in range(10):
    total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 0, True)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))
#training
for i in range(500):
    total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 1, True)
    print(total_reward)
    total_rewards.append(total_reward)
    if i%50 == 0:
        agent.save('saves/data.agent')
agent.save('saves/data.agent')
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))
total_rewards = []
#evaluating
for i in range(10):
    total_reward, steps = agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, 0, True)
    print(total_reward)
    total_rewards.append(total_reward)
print(total_rewards)
print('mean: ' + str(np.array(total_rewards).mean()))