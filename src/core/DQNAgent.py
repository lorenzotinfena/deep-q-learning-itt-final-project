
import gym
from gym import wrappers
from core.Net import Net
import random
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
		self.nn = Net(env.observation_space.shape[0], env.action_space.n, 0.5)
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
		current_state = current_state.reshape(1, -1)
		while not done:
			# choose action
			q_values = self.nn.model.predict(current_state)
			action = self.env.action_space.sample()  if random.uniform(0, 1) < exploration_epsilon else np.argmax(q_values)

			next_state, reward, done, _ = self.env.step(action)
			next_state = next_state.reshape(1, -1)
			if monitor:
				total_reward += reward
				steps += 1
				self.env.render()
			
			# find target q(s)
			q_values_target = np.copy(q_values)
			q_values_target[0][action] = float = reward + discount_factor * np.max(self.nn.model.predict(next_state))

			# update neural network
			self.nn.model.fit(current_state, q_values_target, epochs=1, verbose=0)

			# set current state
			current_state = next_state


		if monitor: return total_reward, steps