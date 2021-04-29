import gym
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
env.close()