import gym
from gym import wrappers
from core.NeuralNetwork import NeuralNetwork
import random
import numpy as np
import pickle as pk
class DQNAgent:
	""" Deep Q learning agent
	"""
	def __init__(self, env: gym.Env, nn: NeuralNetwork):
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
		self.nn = nn
	
	def start_episode(self, discount_factor, learning_rate, exploration_epsilon: float = 0):
		""" start the episode, finish when enviroment return done=True
			Use epsilon-greedy algorithm to
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			exploration_epsilon: exploration probability
		"""
		# get the first state
		current_state = self.env.reset()

		done = False
		while not done:
			# choose action
			a, z = self.nn.forward_propagate(current_state)
			q_values_predicted = a[-1]
			action = self.env.action_space.sample()  if np.random.uniform(0, 1) < exploration_epsilon else np.argmax(q_values_predicted)
			
			# execute action
			next_state, reward, done, _ = self.env.step(action)
			
			# find target q(s)
			q_values_target = np.copy(q_values_predicted)
			if done: q_values_target[action] = reward
			else: q_values_target[action] = reward + discount_factor * np.max(self.nn.predict(next_state))

			# update neural network
			self.nn.backpropagate(z, a, q_values_target, learning_rate)

			# set current state
			current_state = next_state
		
	def start_episode_and_evaluate(self, discount_factor: float, learning_rate: float, exploration_epsilon: float = 0, render=False, optimize=True):
		""" start the episode, finish when enviroment return done=True
			Use epsilon-greedy algorithm to 
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			exploration_epsilon: exploration probability
			render: if env is rendered at each step
			optimize: if nn optimization in saved after this episode
		"""
		if optimize:
			# backup weights
			original_weights = self.nn.weights
		
		# initialize metrics
		total_reward = 0
		steps = 0
		costs = []

		# get the first state
		current_state = self.env.reset()

		done = False
		while not done:
			# choose action
			a, z = self.nn.forward_propagate(current_state)
			q_values_predicted = a[-1]
			action = self.env.action_space.sample()  if np.random.uniform(0, 1) < exploration_epsilon else np.argmax(q_values_predicted)

			# render
			if render: self.env.render()
			
			# execute action
			next_state, reward, done, _ = self.env.step(action)
			
			# find target q(s)
			q_values_target = np.copy(q_values_predicted)
			if done: q_values_target[action] = reward
			else: q_values_target[action] = reward + discount_factor * np.max(self.nn.predict(next_state))
			
			# update monitor metrics
			total_reward += reward
			steps += 1
			costs.append(self.nn.cost_function(q_values_predicted, q_values_target))
			
			# update neural network
			self.nn.backpropagate(z, a, q_values_target, learning_rate)

			# set current state
			current_state = next_state

		# restore original weights
		if optimize:
			self.nn.weights = original_weights

		return total_reward, steps, np.mean(costs)