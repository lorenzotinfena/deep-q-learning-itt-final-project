from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch
from tqdm import tqdm

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()


env = gym.make("CartPole-v0")
env = gym.wrappers.Monitor(env, "recording", force=True)

observation = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
env.close()
env.env.close()