import gym
from IPython import display
import matplotlib.pyplot as plt
#%matplotlib inline
import gym 
import pybulletgym

import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1920, 1080))
_ = _display.start()

ids = ['InvertedPendulumPyBulletEnv-v0',
'InvertedDoublePendulumPyBulletEnv-v0',
'InvertedPendulumSwingupPyBulletEnv-v0',
'ReacherPyBulletEnv-v0',
'PusherPyBulletEnv-v0',
'ThrowerPyBulletEnv-v0',
'StrikerPyBulletEnv-v0',
'Walker2DPyBulletEnv-v0',
'HalfCheetahPyBulletEnv-v0',
'AntPyBulletEnv-v0',
'HopperPyBulletEnv-v0',
'HumanoidPyBulletEnv-v0',
'HumanoidFlagrunPyBulletEnv-v0',
'HumanoidFlagrunHarderPyBulletEnv-v0',
'AtlasPyBulletEnv-v0',
'InvertedPendulumMuJoCoEnv-v0',
'InvertedDoublePendulumMuJoCoEnv-v0',
'Walker2DMuJoCoEnv-v0',
'HalfCheetahMuJoCoEnv-v0',
'AntMuJoCoEnv-v0',
'HopperMuJoCoEnv-v0',
'HumanoidMuJoCoEnv-v0']
for id in ids:
    env = gym.make(id)
    print(id + '\t' + str(env.action_space))
    print('--------------------')
exit()
env = gym.wrappers.Monitor(env, "./recording",force=True)
for _ in range(2):
    env.reset()
    done = False
    while not done:
        env.render()#mode='rgb_array')
        _, _, done, _ = env.step(env.action_space.sample())
env.close()

