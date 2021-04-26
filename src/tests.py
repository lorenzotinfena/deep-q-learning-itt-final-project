from core.DQNAgent import DQNAgent
import gym
import numpy as np
import torch
from tqdm import tqdm

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

env = gym.make("CartPole-v1")
env = gym.wrappers.Monitor(env, 'recording', force=True)
for i in range(50):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        _, _, done, _ = env.step(env.action_space.sample())
    env.close()