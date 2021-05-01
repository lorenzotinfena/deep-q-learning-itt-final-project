import gym
import numpy as np
from collections import deque
import warnings

env=gym.make('CartPole-v1')

def relu(mat):
    return np.multiply(mat,(mat>0))
    
def relu_derivative(mat):
    return (mat>0)*1

class NNLayer:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(output_size, input_size)).T

        self.activation_function = activation
        self.lr = lr

    # Compute the forward pass for this layer
    def forward(self, inputs, remember_for_backprop=True):
        input_with_bias = np.append(inputs,1)
        unactivated = np.dot(input_with_bias, self.weights)
        output = unactivated
        if self.activation_function != None:
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output    
        
    def update_weights(self, gradient):
        self.weights -= self.lr*gradient
        
    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out),gradient_from_above)
        
        D_i = self.backward_store_in.reshape(-1, 1).dot(adjusted_mul.reshape(1, -1))
        #print(np.sum(D_i))
        delta_i = self.weights.dot(adjusted_mul)[:-1]
        self.update_weights(D_i)
        return delta_i
        
class RLAgent:
    # class representing a reinforcement learning agent
    env = None
    def __init__(self, env):
        self.env = env
        self.hidden_size = 5
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.num_hidden_layers = 2
        self.gamma = 0.95
        
        self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
        for i in range(self.num_hidden_layers-1):
            self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
        self.layers.append(NNLayer(self.hidden_size+1, self.output_size))
        
    def select_action(self, observation):
        values = self.forward(np.asmatrix(observation))
        if (np.random.random() > 0.5):
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)
            
    def forward(self, observation, remember_for_backprop=True):
        vals = np.copy(observation)
        index = 0
        for layer in self.layers:
            vals = layer.forward(vals, remember_for_backprop)
            index = index + 1
        return vals
        
    def update(self, done, action_selected, new_obs, prev_obs):
        action_values = self.forward(prev_obs, remember_for_backprop=True)
        next_action_values = self.forward(new_obs, remember_for_backprop=False)
        experimental_values = np.copy(action_values)
        if done:
            experimental_values[action_selected] = -1
        else:
            experimental_values[action_selected] = 1 + self.gamma*np.max(next_action_values)
        self.backward(action_values, experimental_values)
        
    def backward(self, calculated_values, experimental_values): 
        # values are batched = batch_size x output_size
        delta = (calculated_values - experimental_values)
        # print('delta = {}'.format(delta))
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
                

np.random.seed(1000)

# Global variables
NUM_EPISODES = 10000
MAX_TIMESTEPS = 1000
model = RLAgent(env)

def main():
    # The main program loop
    for i_episode in range(NUM_EPISODES):
        env.seed(0)
        observation = env.reset()
        # Iterating through time steps within an episode
        for t in range(MAX_TIMESTEPS):
            action = model.select_action(observation)
            prev_obs = observation
            observation, reward, done, info = env.step(action)

            # Keep a store of the agent's experiences
            model.update(done, action, observation, prev_obs)
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                print('Episode {} ended after {} timesteps'.format(i_episode, t+1))
                break
if __name__ == '__main__':
    main()