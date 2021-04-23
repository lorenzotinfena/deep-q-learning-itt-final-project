from core.Agent import Agent
import gym
import numpy as np
import torch


env = gym.make("CartPole-v1")
print('state_space / observation_space: ' + str(np.array(env.observation_space)))
print('action_space: ' + str(env.action_space))
agent = Agent(env)