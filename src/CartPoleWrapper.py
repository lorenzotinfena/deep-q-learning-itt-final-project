import gym

class CartPoleWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, -1 if done else reward, done, info