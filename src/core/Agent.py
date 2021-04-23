
import gym
from gym import wrappers
import numpy as np


class DQNAgent:
	""" Deep Q learning agent with
	"""
	def __init__(self, env):
		"""
		Args:
			env: enviroment to use
				is assumed that implements these methods:
				- reset() return first_observation
				- step(action) return next_observation, reward, is_done, info
				- close()
				- render(mode='human')
				and action_space is discrete:------------------------
		"""
		self.env = env
	def start_episode(self, discount_factor: float, learning_rate: float, exploration_epsilon: float = 0, monitor=False):
		""" start the episode, finish when enviroment return done=True
			Use epsilon-greedy algorithm to 
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			exploration_epsilon: exploration probability
		Return:
			if minitor=True return the total reward and the total steps
		"""
		done = False
		if monitor:
			total_reward = 0
			steps = 0

		# get the first state
		current_state = self.env.reset()

		while not done:
			# choose action
			action = self.env.action_space.sample() if random.uniform(0, 1) < exploration_probability else np.argmax(self.nn.forward(current_state))

			next_state, reward, done, _ = self.env.step(action)
			if monitor:
				total_reward += reward
				steps += 1
			#update the


		if monitor: return total_reward, steps