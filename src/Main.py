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

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

def plot_videos(videos_path='.', output_file_path='.'):
	stringa = 'ffmpeg -i \"concat:'
	elenco_video = glob.glob(f'{videos_path}/*.mp4')
	if len(elenco_video) == 0:
		print('0 mp4 found in this path')
		return
	elenco_file_temp = []
	for f in elenco_video:
		file = videos_path + '/temp' + str(elenco_video.index(f) + 1) + '.ts'
		os.system('ffmpeg -i ' + f + ' -c copy -bsf:v h264_mp4toannexb -f mpegts ' + file)
		elenco_file_temp.append(file)
	for f in elenco_file_temp:
		stringa += f
		if elenco_file_temp.index(f) != len(elenco_file_temp)-1:
			stringa += '|'
		else:
			stringa += f'\" -c copy -y -bsf:a aac_adtstoasc {output_file_path}'
	os.system(stringa)
	display(Video(output_file_path))

def plot_metrics(n_episodes, total_rewards, number_steps):
    count = len(number_steps)
    _n_episodes = n_episodes[:count].copy()
    _total_rewards = total_rewards[:count].copy()
    _number_steps = number_steps.copy()
    
    cycol = cycle('bgrcmk')
    f, (ax1, ax2) = plt.subplots(1, 2)

    samples = 20
    _n_episodes.extend([np.array(_n_episodes[-(count%samples):]).mean()] * (samples - count%samples))
    _n_episodes = np.array(_n_episodes).reshape(-1, samples).mean(axis=1)
    
    _total_rewards.extend([np.array(_total_rewards[-(count%samples):]).mean()] * (samples - count%samples))
    _total_rewards = np.array(_total_rewards).reshape(-1, samples).mean(axis=1)
    
    _number_steps.extend([np.array(_number_steps[-(count%samples):]).mean()] * (samples - count%samples))
    _number_steps = np.array(_number_steps).reshape(-1, samples).mean(axis=1)

    ax1.set_xlabel('episodes')
    ax1.set_ylabel('_total_rewards')
    ax1.plot(_n_episodes, _total_rewards, c=next(cycol))
    
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('number_steps')
    ax2.plot(_n_episodes, _number_steps, c=next(cycol))
    
    f.tight_layout()

# %% [markdown]
# Initialize deep Q-learning agent, neural network, and parameters
# %%
#np.random.seed(1000)
agent = DQNAgent(env=CartPoleWrapper(gym.make("CartPole-v1")),
				nn=CartPoleNeuralNetwork(), replay_memory_max_size=250, batch_size=30)

DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.001

n_episodes = []
total_rewards = []
number_steps = []
total_episodes = 0


# %% [markdown]
# Training
# %%
while total_episodes <= 1000:
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0, min_epsilon=0, render=False, optimize=False)
    print(f'\ntotal_episodes_training: {total_episodes}\tsteps: {steps}\ttotal_reward: {total_reward}', flush = True)
    n_episodes.append(total_episodes)
    total_rewards.append(total_reward)
    number_steps.append(steps)

    for i in tqdm(range(20), 'learning...'):
        agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 1, epsilon_decay=0.99, min_epsilon=0.01, render=False, optimize=True)
    total_episodes += i+1

    if total_episodes % 200 == 0:
        agent.save_weights(f'saves/data{total_episodes}.nn')


# %% [markdown]
# Visualize training metrics
# %%
plot_metrics(n_episodes, total_rewards, number_steps)


# %% [markdown]
# Evaluating
# %%
agent.env = gym.wrappers.Monitor(agent.env, 'recording/tmp-videos', force=True, video_callable=lambda episode_id: True)
#agent.load_weights('saves/data396000.nn')

for i in range(10):
    total_reward, steps = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0, render=True, optimize=False)
    print(f'{i}\t{steps}\t{total_reward}')
agent.env.close()

agent.env = agent.env.env
plot_videos('recording/tmp-videos', f'recording/ouuuuuuuuu-episodes.mp4')


# %% [markdown]
evaluation

train 1 000 (with %10 evaluation)

evaluation

train 9 000 - 10 000 (with %90 evaluation)

evaluation

train 90 000 - 100 000

evaluation

train 900 000 - 1 000 0000

evaluation1