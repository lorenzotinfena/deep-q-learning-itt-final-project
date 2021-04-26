import gym
from gym import wrappers
from core.NeuralNetwork import NeuralNetwork
import random
import numpy as np
import pickle as pk

class DQNAgent:
	""" Deep Q learning agent
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
		# activation function:
		#def sigmoid(x: np.array, derivative=False):
		#	if derivative:
		#		return np.exp(-x) / (1 + np.exp(-x)) ** 2
		#	return 1 / (1 + np.exp(-x))
		# cost function:
		def sum_square_error(predicted, target):
			return 1/2 * np.sum((predicted - target)**2)
		def sum_square_error_derivative(predicted, target):
			return predicted - target
		def identity(x, derivative=False):
			if derivative: return np.ones(len(x))
			else: return x
		self.cost_function = sum_square_error
		cost_function_derivative = sum_square_error_derivative
		activation_functions = [identity]*3

		self.env = env
		self.nn = NeuralNetwork([env.observation_space.shape[0], 5, 5, env.action_space.n], activation_functions, cost_function_derivative)
	
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
			a, z = self.nn.forward_propagate(current_state)
			q_values_predicted = a[-1]
			action = self.env.action_space.sample()  if np.random.uniform(0, 1) < exploration_epsilon else np.argmax(q_values_predicted)

			next_state, reward, done, _ = self.env.step(action)
			
			# find target q(s)
			q_values_target = np.copy(q_values_predicted)
			q_values_target[action]: float = reward + discount_factor * np.max(self.nn.predict(next_state))

			if monitor:
				total_reward += reward
				steps += 1
				self.env.render()
				if (np.isnan(self.cost_function(q_values_predicted, q_values_target))):
					fddsf = 1
				print(str(self.cost_function(q_values_predicted, q_values_target)))

			# update neural network
			self.nn.backpropagate(a, z, q_values_target, learning_rate)

			# set current state
			current_state = next_state


		if monitor: return total_reward, steps

	
	def save(self, path: str):
		with open(path, "wb") as file:
			pk.dump((self.nn.weights, self.nn.biases), file)

	@staticmethod
	def load(path: str):
		with open(path, "rb") as file:
			return pk.load(file)