import gym
import numpy as np
class CartPoleWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward -= np.abs(observation[0] / 10)
        return observation, -10 if done else reward, done, info