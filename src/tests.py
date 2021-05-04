from core.DQNAgent import DQNAgent
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
from CartPoleWrapper import CartPoleWrapper


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
      stringa += f'\" -c copy  -bsf:a aac_adtstoasc {output_file_path}'
  os.system(stringa)
  display(Video(output_file_path))

def plot_metrics():
    cycol = cycle('bgrcmk')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_xlabel('episodes')
    ax1.set_ylabel('total_rewards')
    ax1.plot(train_episode_indexes, total_rewards, c=next(cycol))

    ax2.set_xlabel('episodes')
    ax2.set_ylabel('number_steps')
    ax2.plot(train_episode_indexes, number_steps, c=next(cycol))

    ax3.set_xlabel('episodes')
    ax3.set_ylabel('cost_function_means')
    ax3.plot(train_episode_indexes, cost_function_means, c=next(cycol))

    f.tight_layout()



env = CartPoleWrapper(gym.make("CartPole-v1"))
env = gym.wrappers.Monitor(env, 'recording/tmp-videos', force=True, video_callable=lambda episode_id: True)
agent = DQNAgent(env)



DISCOUNT_FACTOR = 0.95
LEARNING_RATE = 0.0002
total_rewards = []
number_steps = []
cost_function_means = []
train_episode_indexes = []


for i in range(100):
  total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 1, render=False, optimize=True)
  print(f'{steps}\t{mean_cost_function}')
  total_rewards.append(total_reward)
  number_steps.append(steps)
  cost_function_means.append(mean_cost_function)
  train_episode_indexes.append(i)

print('FATTO')
for i in range(1000):
  total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0, render=False, optimize=True)
  print(f'{steps}\t{mean_cost_function}')
  total_rewards.append(total_reward)
  number_steps.append(steps)
  cost_function_means.append(mean_cost_function)
  train_episode_indexes.append(i)
agent.save('saves/data1.nn')


# %%
from core.DQNAgent import DQNAgent
from cartpole.CartPoleNeuralNetwork import CartPoleNeuralNetwork
from cartpole.CartPoleWrapper import CartPoleWrapper
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

def plot_metrics():
    cycol = cycle('bgrcmk')
    f, (ax1, ax2) = plt.subplots(1, 2)

    samples = 20

    n_episodes.append([np.array(n_episodes[-(n_episodes%samples):]).mean()] * (n_episodes/samples - n_episodes%samples))
    n_episodes = np.array(n_episodes).reshape(samples, -1).mean(axis=1)

    total_rewards.append([np.array(total_rewards[-(total_rewards%samples):]).mean()] * (total_rewards/samples - total_rewards%samples))
    total_rewards = np.array(total_rewards).reshape(samples, -1).mean(axis=1)

    number_steps.append([np.array(number_steps[-(number_steps%samples):]).mean()] * (number_steps/samples - number_steps%samples))
    number_steps = np.array(number_steps).reshape(samples, -1).mean(axis=1)

   	ax1.set_xlabel('episodes')
    ax1.set_ylabel('total_rewards')
    ax1.plot(n_episodes, total_rewards, c=next(cycol))
    
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('number_steps')
    ax2.plot(n_episodes, number_steps, c=next(cycol))
    
    f.tight_layout()