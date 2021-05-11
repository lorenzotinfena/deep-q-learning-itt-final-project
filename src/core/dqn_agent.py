import pickle as pk

import gym
import numpy as np
from gym import wrappers

from core.neural_network import NeuralNetwork
from core.replay_memory import ReplayMemory
from copy import copy
class DQNAgent:
	""" Deep Q learning agent
	"""
	def __init__(self, env: gym.Env, nn: NeuralNetwork, replay_memory_max_size, batch_size):
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
		self._nn = nn
		self._target_nn = copy(self._nn)
		self._sync_target_nn_weights()
		self._replay_memory = ReplayMemory(max_size=replay_memory_max_size)
		self._batch_size = batch_size
	
	def load_weights(self, path):
		self._nn.load_weights(path)
		
	def save_weights(self, path):
		self._nn.save_weights(path)
  
	def _sync_target_nn_weights(self):
		self._target_nn.weights = self._nn.clone_weights()
	
	def start_episode(self, discount_factor, learning_rate,
                                epsilon, epsilon_decay = 0.99, min_epsilon = 0.01, momentum=0.9):
		""" start the episode, finish when enviroment return done=True
			Use epsilon-greedy algorithm to 
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			epsilon: exploration epsilon
			epsilon_decay: epsilon decrease factor at every optimization
			min_epsilon: minimum epsilon value
			render: if env is rendered at each step
			optimize: if _nn optimization in saved after this episode
		"""
		# get the first state
		state = self.env.reset()
		steps = 0
		done = False
		while not done:
			# choose action
			if np.random.uniform(0, 1) < epsilon:
				action = self.env.action_space.sample()
			else:
				action = np.argmax(self._nn.predict(state))
			
			# execute action
			next_state, reward, done, _ = self.env.step(action)
			
			# store experience
			self._replay_memory.put(state, action, reward, done, next_state)
			steps += 1
			if len(self._replay_memory) >= self._batch_size:
				# get experience batch from replay memory
				for state_exp, action_exp, reward_exp, done_exp, next_state_exp in self._replay_memory.get(batch_size=self._batch_size):
					# find target q(s)
					a, z = self._nn.forward_propagate(state_exp)
					q_values_target = np.copy(a[-1])
					if done: q_values_target[action_exp] = reward_exp
					else: q_values_target[action_exp] = reward_exp + discount_factor * np.max(self._target_nn.predict(next_state_exp))
					
					# update neural network
					self._nn.backpropagate(z, a, q_values_target, learning_rate, momentum)
				
				if steps % 10 == 0:
					# sync target nn weights
					self._sync_target_nn_weights()
    
				# epsilon-decay algorithm
				epsilon *= epsilon_decay
				if epsilon < min_epsilon:
					epsilon = min_epsilon

			# set current state
			state = next_state
		
		self._sync_target_nn_weights()

	def start_episode_and_evaluate(self, discount_factor, learning_rate,
                                epsilon, epsilon_decay = 0.99, min_epsilon = 0.01, momentum=0.9, render=False, optimize=True):
		""" start the episode, finish when enviroment return done=True
			Use epsilon-greedy algorithm to 
		Args:
			discount_factor: how much the immediate rewards matter than future rewards
				0 <= discount_factor <= 1
			learning_rate: with how much strength you want to learn
				0 < learning_rate
			epsilon: exploration epsilon
			epsilon_decay: epsilon decrease factor at every optimization
			min_epsilon: minimum epsilon value
			render: if env is rendered at each step
			optimize: if _nn optimization in saved after this episode
		"""
		if not optimize:
			# backup weights
			original_weights = self._nn.clone_weights()
			original_v = self._nn.clone_v()
		
		# initialize metrics
		total_reward = 0
		steps = 0

		# get the first state
		state = self.env.reset()

		done = False
		while not done:
			# choose action
			if np.random.uniform(0, 1) < epsilon:
				action = self.env.action_space.sample()
			else:
				action = np.argmax(self._nn.predict(state))

			# render
			if render: self.env.render()
			
			# execute action
			next_state, reward, done, _ = self.env.step(action)

			# update monitor metrics
			total_reward += reward
			steps += 1
			
			# store experience
			if optimize: self._replay_memory.put(state, action, reward, done, next_state)
			
			if len(self._replay_memory) >= self._batch_size:
				# get experience batch from replay memory
				for state_exp, action_exp, reward_exp, done_exp, next_state_exp in self._replay_memory.get(batch_size=self._batch_size):
					# find target q(s)
					a, z = self._nn.forward_propagate(state_exp)
					q_values_target = np.copy(a[-1])
					if done: q_values_target[action_exp] = reward_exp
					else: q_values_target[action_exp] = reward_exp + discount_factor * np.max(self._target_nn.predict(next_state_exp))
					
					# update neural network
					self._nn.backpropagate(z, a, q_values_target, learning_rate, momentum)
				
				if steps % 10 == 0:
					# sync target nn weights
					self._sync_target_nn_weights()
    
				# epsilon-decay algorithm
				epsilon *= epsilon_decay
				if epsilon < min_epsilon:
					epsilon = min_epsilon

			# set current state
			state = next_state

		self._sync_target_nn_weights()

		if not optimize:
			# restore original weights
			self._nn.weights = original_weights
			self._sync_target_nn_weights()
			self._nn.v = original_v

		return total_reward, steps
	

	def __plot_cost_function_changing_weight(self, l, i, j, input, target, minn=-1000, maxx=1000, step=0.1):
		import matplotlib.pyplot as plt

		_nn_mtp = copy(self._nn)
		_nn_mtp.weights = self._nn.clone_weights()

		y = []
		x = []
		for z in np.arange(minn, maxx, step):
			_nn_mtp.weights[l][i][j] = z
			x.append(z)
			y.append(_nn_mtp.cost_function(_nn_mtp.predict(input), target))
		
		fig, ax = plt.subplots()
		ax.plot(x, y)
		ax.grid()
		plt.show()