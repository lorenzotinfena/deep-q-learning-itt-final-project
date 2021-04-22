import gym
from gym import wrappers

from Environment import Environment

print(env.close())
class Agent:
	def __init__(self, env):
		"""
		Args:
			env: enviroment to use
				is assumed that implements these methods:
				- reset() return first_observation
				- step(action) return next_observation, reward, is_done, info
				- close()
				- render(mode='human')
		"""
		self.env = env
	def start_episode(self, discount_factor: float, learning_rate: float, exploration_factor: float = 0, monitor=False):
		""" start the episode, finish when enviroment return done=True
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			exploration_factor: 
		"""
	