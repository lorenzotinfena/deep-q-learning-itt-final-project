# %%
from core.dqn_agent import DQNAgent
from cartpole.cartpole_neural_network import CartPoleNeuralNetwork
from cartpole.cartpole_wrapper import CartPoleWrapper
import gym
import numpy as np
import torch
from tqdm import tqdm
import glob
import os
from IPython.display import Video
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import sys
from cartpole.cartpole_neural_network import CartPoleNeuralNetwork
from copy import *


a = CartPoleNeuralNetwork()
b = copy(a)
b.weights = deepcopy(a.weights)
a.weights[0][0][0] = 1



agent = DQNAgent(env=CartPoleWrapper(gym.make("CartPole-v1")),
				nn=CartPoleNeuralNetwork(), replay_memory_max_size=250, batch_size=30)

DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.001

n_episodes = []
total_rewards = []
number_steps = []
total_episodes = 0



while total_episodes <= 1000:
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=0, min_epsilon=0, render=False, optimize=False)
    print(f'\ntotal_episodes_training: {total_episodes}\tsteps: {steps}\ttotal_reward: {total_reward}', flush = True)
    n_episodes.append(total_episodes)
    total_rewards.append(total_reward)
    number_steps.append(steps)

    for i in tqdm(range(50), 'learning...'):
        agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=1, epsilon_decay=0.99, min_epsilon=0.01, render=False, optimize=True)
    total_episodes += i+1

    if total_episodes % 100 == 0:
        agent.save_weights(f'saves/data{total_episodes}.nn')