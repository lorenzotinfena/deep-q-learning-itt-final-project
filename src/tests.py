from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch
from tqdm import tqdm

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

#env = gym.wrappers.Monitor(env, 'recording', force=True)
np.random.seed(1000)

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00005
total_rewards = []
number_steps = []
cost_function_means = []
env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording', force=True, video_callable=lambda episode_id: True)
agent = DQNAgent(env)

for i in range(10):
    total_reward, steps, mean_cost_function = agent.start_episode_and_evaluate(DISCOUNT_FACTOR, LEARNING_RATE, 0, render=True, optimize=False)
    total_rewards.append(total_reward)
    total_rewards.append(total_reward)
    total_rewards.append(total_reward)
env.close()

'''import gym
from IPython import display
import matplotlib.pyplot as plt
#%matplotlib inline
import gym 

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1920, 1080))
_ = _display.start()


env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, "./recording", force=True)
for _ in range(2):
    env.reset()
    done = False
    while not done:
        env.render()
        _, _, done, _ = env.step(env.action_space.sample())
env.close()'''