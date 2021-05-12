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
import shutil
from pathlib import Path
import shutil
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils import *
import plotly.express as px

# %% [markdown]
# Initialize deep Q-learning agent and neural network
# %%
seed = 1000
np.random.seed(seed)
agent = DQNAgent(env=CartPoleWrapper(gym.make("CartPole-v1")),
                nn=CartPoleNeuralNetwork(), replay_memory_max_size=10000, batch_size=30)
#agent.env.seed(0)
agent.env.action_space.np_random.seed(seed)

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.0001
STEPS_TO_SYNC_TARGET_NN=10

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
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, steps_to_sync_target_nn=STEPS_TO_SYNC_TARGET_NN,
    epsilon=0, min_epsilon=0, momentum=0.4, render=False, optimize=False)
    logger.set_description(f'episode: {total_episodes}\tsteps: {steps}\ttotal_reward: {total_reward}')
    n_episodes.append(total_episodes)
    total_rewards.append(total_reward)
    number_steps.append(steps)

    for i in range(10):
        agent.start_episode(DISCOUNT_FACTOR, LEARNING_RATE, steps_to_sync_target_nn=STEPS_TO_SYNC_TARGET_NN,
        epsilon=1, epsilon_decay=0.99, min_epsilon=0.01, momentum=0.4)
    total_episodes += i+1

    if total_episodes % 20 == 0:
        agent.save_weights(f'results/cartpole/saves/data{total_episodes}.nn')
# %% [markdown]
# Visualize training metrics
# %%
plot_metrics(n_episodes, total_rewards, number_steps,)
# %% [markdown]

# Evaluation

## Video demos
# %%
if Path('results/cartpole/recording/tmp-videos').exists():
	shutil.rmtree('results/cartpole/recording/tmp-videos')
agent.env = gym.wrappers.Monitor(agent.env, 'results/cartpole/recording/tmp-videos', force=True, video_callable=lambda episode_id: True)
agent.load_weights('results/cartpole/good-results/3best/saves/data320.nn')

for i in range(10):
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=0, min_epsilon=0, momentum=0.5, render=False, optimize=False)
    print(f'{i}\t{steps}\t{total_reward}')
agent.env.close()

agent.env = agent.env.env

plot_videos('results/cartpole/recording/tmp-videos', f'results/cartpole/recording/output.mp4')

# %% [markdown]
## Compute total reward distribution
# %%
agent.load_weights('results/cartpole/good-results/3best/saves/data320.nn')
total_rewards = []
number_steps = []
for i in range(10000):
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, epsilon=0, min_epsilon=0, momentum=0.5, render=False, optimize=False)
    if i % 100 == 0: print(f'{i}\t{steps}\t{total_reward}')
    total_rewards.append(total_reward)
    number_steps.append(steps)
# %% [markdown]
## Plot total reward distribution
# %%
x = total_rewards
fig = px.histogram(x=x, nbins=round(len(x)/10))
fig.update_xaxes(title_text='total_rewards')
fig.add_vline(x=np.mean(x), line_width=3, line_dash="dash", line_color="green")
fig.show()
print(f'mean: {np.mean(x)}')
print(f'standard deviation: {np.std(x)}')