# %%
from core.dqn_agent import DQNAgent
from cartpole.cartpole_neural_network import CartPoleNeuralNetwork
from cartpole.cartpole_wrapper import CartPoleWrapper
from spaceinvaders.space_invaders_neural_network import SpaceInvadersNeuralNetwork
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
import shutil
from pathlib import Path
import shutil
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import *

# %% [markdown]
# Initialize deep Q-learning agent, neural network, and parameters
# %%
seed = 1000
np.random.seed(seed)
agent = DQNAgent(env=CartPoleWrapper(gym.make("CartPole-v1")),
                nn=CartPoleNeuralNetwork(), replay_memory_max_size=10000, batch_size=20)
agent.env.seed(0)
agent.env.action_space.np_random.seed(seed)

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001

n_episodes = []
total_rewards = []
number_steps = []
total_episodes = 0


# %% [markdown]
# Training
# %%
if Path('results/cartpole/saves').exists():
    shutil.rmtree('results/cartpole/saves')

logger = tqdm(range(100))
for _ in logger:
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=0, min_epsilon=0, momentum=0.4, render=False, optimize=False)
    logger.set_description(f'episode: {total_episodes}\tsteps: {steps}\ttotal_reward: {total_reward}')
    n_episodes.append(total_episodes)
    total_rewards.append(total_reward)
    number_steps.append(steps)

    for i in range(10):
        agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=1, epsilon_decay=0.99, min_epsilon=0.01, momentum=0.4)
    total_episodes += i+1

    if total_episodes % 20 == 0:
        agent.save_weights(f'results/cartpole/saves/data{total_episodes}.nn')
# %% [markdown]
# Visualize training metrics
# %%
plot_metrics(n_episodes, total_rewards, number_steps, -1)
plot_metrics(n_episodes, total_rewards, number_steps, 5)
# %% [markdown]
# Evaluating


# %%
if Path('results/cartpole/recording/tmp-videos').exists():
	shutil.rmtree('results/cartpole/recording/tmp-videos')
agent.env = gym.wrappers.Monitor(agent.env, 'results/cartpole/recording/tmp-videos', force=True, video_callable=lambda episode_id: True)
agent.load_weights('results/cartpole/saves/data200.nn')

for i in range(10):
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=0, min_epsilon=0, momentum=0.5, render=False, optimize=False)
    print(f'{i}\t{steps}\t{total_reward}')
agent.env.close()

agent.env = agent.env.env


plot_videos('results/cartpole/recording/tmp-videos', f'results/cartpole/recording/output.mp4')

# %%
np.random.seed(500)
agent = DQNAgent(env=gym.make("SpaceInvaders-ram-v0"),
				nn=SpaceInvadersNeuralNetwork(), replay_memory_max_size=10000, batch_size=10)
agent.env.seed(500)

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0

n_episodes = []
total_rewards = []
number_steps = []
total_episodes = 0
# %% [markdown]
# Training
# %%
if Path('saves/spaceinvaders').exists():
	shutil.rmtree('saves/spaceinvaders')
 
logger = tqdm(range(100))
for _ in logger:
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=1, epsilon_decay=0.995, min_epsilon=0.01, render=False, optimize=True)
    logger.set_description(f'episode: {total_episodes}\tsteps: {steps}\ttotal_reward: {total_reward}')
    n_episodes.append(total_episodes)
    total_rewards.append(total_reward)
    number_steps.append(steps)

    for i in range(0):
        agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=1, epsilon_decay=0.995, min_epsilon=0.01)
    total_episodes += i+1

    if total_episodes % 50 == 0:
        agent.save_weights(f'saves/spaceinvaders/data{total_episodes}.nn')

#%%
env=retro.make(game='SpaceInvaders-Atari2600')
env.reset()
done = False
while not done:
    ne, re, done, _ = env.step(env.action_space.sample())
    print(re)