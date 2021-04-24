import numpy as np
import pickle as pk

class NeuralNetwork:
    def __init__(self, n_neurons: np.array, activation_functions: np.array, cost_function_derivative):
        """
		Args:
			n_neurons: np.array
                number of neurons per layer
            activation_functions: np.array[len(n_neurons)-1] dtype:function(np.array, derivative=False)->np.array
            cost_function_derivative: function(predicted: np.array, target: np.array)->np.array
                partial derivative of cost_function respect to predicted values
		"""
        self.activation_functions = activation_functions
        self.cost_function_derivative = cost_function_derivative
        self.bias = [(np.random.randn(layer)) for layer in n_neurons[1:]]
        self.n_neurons = n_neurons
        self.weights = [(np.random.randn(n_neurons[i + 1], n_neurons[i])) for i in range(len(n_neurons) - 1)]

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
		Args:
			input: np.ndarray 1dim
                array that rapresent inputs
		Return:
			prediction: np.ndarray 1dim
		"""
        for thetas, bias, activation_function in zip(self.thetas, self.bias,self.activation_functions):
            input = self.activation_function(thetas.dot(input) + bias)
        return input.T[0]

    def forward_propagate(self, input: np.ndarray) -> (np.ndarray,  np.ndarray):
        """
        Args:
            input: np.ndarray 1dim
                array that rapresent inputs
        Return: predict values
            a, z
        """
        a = [input]
        z = [input]
        for thetas, bias, activation_function in zip(self.thetas, self.bias, self.activation_functions):
            z_ = thetas.dot(input) + bias
            z.append(z_)
            input = self.activation_function(z_)
            a.append(input)
        return np.array(a), np.array(z)

    def backpropagate(self, z: np.ndarray, a: np.ndarray, target_output: np.array, learning_rate: float):
        """
        Args:
            z: np.ndarray
            a: np.ndarray
            target_output: np.array
            learning_rate: float
        """
        gradients_z = self.cost_function_derivative(a[-1], target_output) * self.activation_functions[self.n_neurons[-1]](z[-1], True) # error*activation'()
        for i, (weights, bias, z, a, activation_function) in reversed(enumerate(zip(self.weights, self.bias, z[:-1], a[:-1], self.activation_functions))):
            bias += learning_rate * gradients_z
            if i != 0:
                gradients_a = weights.T.dot(gradients_z)
            weights += learning_rate * gradients_z.dot(a.T)
            if i != 0:
                gradients_z = gradients_a * self.activation_function(z)