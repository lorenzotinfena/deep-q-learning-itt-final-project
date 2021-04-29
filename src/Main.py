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



env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording/tmp-videos', force=True, video_callable=lambda episode_id: True)
agent = DQNAgent(env)

agent.nn.weights[0] = np.array([[ 0.15358959, -0.38499306,  0.45028286, -0.0178086 ,  0.37247454],
 [-0.28766732, -0.45929038 ,-0.10280554, -0.2668678 ,  0.34174072],
 [-0.29291766,  0.24246953 ,-0.10784587, -0.31774348,  0.24353941],
 [-0.43041792 , 0.3853372 ,  0.4526444 ,  0.43114343, -0.08456905]]).T
agent.nn.biases[0] = np.array([-0.47101834  ,0.48202748 ,-0.16036232 , 0.20668719, -0.13812293])

agent.nn.weights[1] = np.array([[-0.4648941  , 0.35505825 , 0.15725351,  0.26568299 , 0.05408724],
 [ 0.38509294 , 0.40419762, -0.4895783 , -0.42544326 ,-0.25537079],
 [-0.36669525 , 0.1979251 , -0.10179512,  0.38312219, -0.31899249],
 [-0.06750083, -0.4818568,   0.19143786 ,-0.03030935 ,-0.37177781],
 [ 0.39133705 , 0.41820362 ,-0.42687901, -0.45455206 ,-0.0614271 ]]).T
agent.nn.biases[1] = np.array([0.10172093, -0.18977297 , 0.18190824, -0.29098685 , 0.0196043])

agent.nn.weights[2] = np.array([[ 0.06598883, -0.05883261],
 [-0.36244384, -0.28645681],
 [-0.36662811 ,-0.1777033 ],
 [-0.26611288,  0.02749816],
 [ 0.06597116, -0.06177472]]).T
agent.nn.biases[2] = np.array( [-0.17812738 , 0.05964081])


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