import numpy as np
import pickle as pk
import gym
import numpy as np
from collections import deque
import warnings

class NeuralNetwork:
    def __init__(self, n_neurons: np.array, path: str = None):
        """
        args:
            n_neurons: np.array
                number of neurons per layer
			path: path to weights and biases dumped
		"""
        self.n_neurons = n_neurons
        if path == None:
            self.biases = [np.random.uniform(low=-0.5, high=0.5, size=(layer)) for layer in n_neurons[1:]]
            self.weights = [np.random.uniform(low=-0.5, high=0.5, size=(n_neurons[i+1], n_neurons[i])) for i in range(len(n_neurons) - 1)]
        else:
            with open(path, "rb") as file:
                self.weights, self.biases = pk.load(file)

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
		Args:
			input: np.ndarray 1dim
                array that rapresent inputs
		Return:
			prediction: np.ndarray 1dim
		"""
        for weights, biases, activation_function in zip(self.weights, self.biases, self.activation_functions):
            input = activation_function(weights.dot(input) + biases)
        return input

    def forward_propagate(self, input: np.array) -> (list[np.ndarray],  list[np.ndarray]):
        """
        Args:
            input: np.ndarray 1dim
                array that rapresent inputs
        Return: predict values
            a, z
        """
        z = [input]
        a = [input]
        for weights, biases, activation_function in zip(self.weights, self.biases, self.activation_functions):
            z_ = weights.dot(input) + biases
            z.append(z_)
            input = activation_function(z_)
            a.append(input)
        return z, a
    
    def backpropagate(self, z: list[np.ndarray], a: list[np.ndarray], target_output: np.array, learning_rate: float):
        """
        Args:
            z: list[np.ndarray]
            a: list[np.ndarray]
            target_output: np.array
            learning_rate: float
        """
        # compute last layer gradients
        gradients_a = self.cost_function_derivative(a[-1], target_output)
        gradients_z = gradients_a * self.activation_functions_derivative[-1](z[-1])
        
        # update weights and biases, and compute gradients for hidden layers
        for weights, biases, z, a, activation_function_derivative in reversed(list(zip(self.weights[1:], self.biases[1:], z[1:-1], a[1:-1], self.activation_functions_derivative[:-1]))):
            weights -= learning_rate * gradients_z.reshape(-1, 1).dot(a.reshape(1, -1))
            biases -= learning_rate * gradients_z
            gradients_a = weights.T.dot(gradients_z)
            gradients_z = gradients_a * activation_function_derivative(z)

        #update first weights and biases
        self.weights[0] -= learning_rate * gradients_z.reshape(-1, 1).dot(a[0].reshape(1, -1))
        self.biases[0] -= learning_rate * gradients_z
    def save(self, path: str):
        with open(path, "wb") as file:
            pk.dump((self.weights, self.biases), file)


class CustomNeuralNetwork(NeuralNetwork):
    def __init__(self, n_neurons: np.array, path: str):
        """
        args:
            n_neurons: np.array
                number of neurons per layer
			path: path to weights and biases dumped
		"""
        super().__init__(n_neurons, path)


        def identity(x): return x
        def identity_derivative(x): return np.ones(len(x))

        def ReLU(x):
            return np.multiply(x,(x>0))
        def ReLU_derivative(x):
            return (x>0)*1

        def sum_square_error(predicted, target):
            return np.sum((predicted - target)**2) / 2
        def sum_square_error_derivative(predicted, target):
            return predicted - target
        
        """
        Define in object functions:
            activation_functions: iter[len(n_neurons)-1] dtype:function(np.array)->np.array
            activation_functions_derivative: iter[len(n_neurons)-1] dtype:function(np.array)->np.array
            cost_function_derivative: function(predicted: np.array, target: np.array)->np.array
                partial derivatives of cost_function respect to predicted values
		"""
        self.activation_functions = [ReLU, ReLU, identity]
        self.activation_functions_derivative = [ReLU_derivative, ReLU_derivative, identity_derivative]
        self.cost_function = sum_square_error
        self.cost_function_derivative = sum_square_error_derivative

env=gym.make('CartPole-v1')

def identity(x): return x
def identity_derivative(x):
    if len(x.shape) != 1:
        return
    ff = np.ones(len(x))
    return np.ones(len(x))

def relu(mat):
    return np.multiply(mat,(mat>0))
    
def relu_derivative(mat):
    return (mat>0)*1

def sum_square_error(predicted, target):
    return np.sum((predicted - target)**2) / 2
def sum_square_error_derivative(predicted, target):
    return predicted - target

class NNLayer:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        print('LAYER')
        print(str(self.weights))
        self.activation_function = activation
        self.lr = lr

    # Compute the forward pass for this layer
    def forward(self, inputs, remember_for_backprop=True):
        # inputs has shape batch_size x layer_input_size 
        input_with_bias = np.append(inputs,1)
        unactivated = np.dot(input_with_bias, self.weights)
        # store variables for backward pass
        output = unactivated
        if self.activation_function != None:
            # assuming here the activation function is relu, this can be made more robust
            output = self.activation_function(output)
        if remember_for_backprop:
            self.backward_store_in = input_with_bias
            self.backward_store_out = np.copy(unactivated)
        return output    
        
    def update_weights(self, gradient):
        self.weights = self.weights - self.lr*gradient
        
    def backward(self, gradient_from_above):
        adjusted_mul = gradient_from_above
        # this is pointwise
        if self.activation_function != None:
            adjusted_mul = np.multiply(relu_derivative(self.backward_store_out),gradient_from_above)
            
        D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1,len(adjusted_mul))))
        delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
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
        
        n_neurons = [4, 5, 5, 2]
        self.weights = []
        for i in range(len(n_neurons) - 1):
            self.weights.append(np.random.uniform(low=-0.5, high=0.5, size=(n_neurons[i+1], n_neurons[i]+1)))
        self.activation_functions = [relu, relu, None]
        self.activation_functions_derivative = [relu_derivative, relu_derivative, None]

        #self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
        #for i in range(self.num_hidden_layers-1):
        #    self.layers.append(NNLayer(self.hidden_size+1, self.hidden_size, activation=relu))
        #self.layers.append(NNLayer(self.hidden_size+1, self.output_size))
        
    def select_action(self, observation):
        #values = self.forward(np.asmatrix(observation))

        z2, a2 = self.forward(observation)
        values = a2[-1]

        if (np.random.random() > 0.5):
            return np.argmax(values)
        else:
            return np.random.randint(self.env.action_space.n)
            
    def forward(self, observation, remember_for_backprop=True):
        input = np.copy(observation)
        z1 = [input]
        a1 = [input]
        for weights, activation_function in zip(self.weights, self.activation_functions):
            z_ = weights.dot(np.append(input, 1))
            z1.append(z_)
            if activation_function != None:
                input = activation_function(z_)
            else: input = z_
            a1.append(input)
        return z1, a1

        #vals = np.copy(observation)
        #index = 0
        #for layer in self.layers:
        #    vals = layer.forward(vals, remember_for_backprop)
        #    index = index + 1
        #return vals
        
    def update(self, done, action_selected, new_obs, prev_obs):
        z3, a3 = self.forward(prev_obs, remember_for_backprop=True)
        action_values = a3[-1]
        z4, a4 = self.forward(new_obs, remember_for_backprop=False)
        next_action_values = a4[-1]
        #action_values = self.forward(prev_obs, remember_for_backprop=True)
        #next_action_values = self.forward(new_obs, remember_for_backprop=False)
        experimental_values = np.copy(action_values)
        if done:
            experimental_values[action_selected] = -1
        else:
            experimental_values[action_selected] = 1 + self.gamma*np.max(next_action_values)
        self.backward(z3, a3, experimental_values)
        #self.backward(action_values, experimental_values)
        
    def backward(self, z3, a3, experimental_values): 
        # values are batched = batch_size x output_size
        #delta = (calculated_values - experimental_values)
        # print('delta = {}'.format(delta))

        try:
            gradients_a = a3[-1] - experimental_values
            y = gradients_a * 34
            #print(y)
        except:
            pass
        if self.activation_functions_derivative[-1] != None:
            gradients_z = gradients_a * self.activation_functions_derivative[-1](z3[-1])
        else: gradients_z = gradients_a
        for weights, z4, a4, activation_function_derivative in reversed(list(zip(self.weights[1:], z3[1:-1], a3[1:-1], self.activation_functions_derivative[:-1]))):
            weights -= LR * gradients_z.reshape(-1, 1).dot(np.append(a4, 1).reshape(1, -1))
            fdf = gradients_z.reshape(-1, 1).dot(np.append(a4, 1).reshape(1, -1)).T
            #print(np.sum(fdf))
            gradients_a = weights.T.dot(gradients_z.reshape(-1, 1)).reshape(-1)[:-1]
            if activation_function_derivative != None:
                gradients_z = gradients_a * activation_function_derivative(z4)
            else: gradients_z = gradients_a
            
        self.weights[0] -= LR * gradients_z.reshape(-1, 1).dot(np.append(a3[0], 1).reshape(1, -1))
        #print(np.sum(gradients_z.reshape(-1, 1).dot(np.append(a3[0], 1).reshape(1, -1)).T))

        #for layer in reversed(self.layers):
        #    delta = layer.backward(delta)
LR = 0.001
                
def main():
    np.random.seed(1000)

    # Global variables
    NUM_EPISODES = 10000
    MAX_TIMESTEPS = 1000
    model = RLAgent(env)


    # The main program loop
    for i_episode in range(NUM_EPISODES):
        observation = env.reset()
        # Iterating through time steps within an episode
        for t in range(MAX_TIMESTEPS):
            action = model.select_action(observation)
            prev_obs = observation
            observation, reward, done, info = env.step(action)
            # Keep a store of the agent's experiences
            if i_episode == 1226:
                fdf = 23
            if any([any([any(np.isnan(weights)) for weights in weights]) for weights in model.weights]):# or any([any(np.isnan(biases)) for biases in model.biases]):
                print('nan weights or biases')
                return
            
            model.update(done, action, observation, prev_obs)
            # epsilon decay
            if done:
                # If the pole has tipped over, end this episode
                print('Episode {} ended after {} timesteps'.format(i_episode, t+1))
                break
if __name__ == '__main__':
    main()