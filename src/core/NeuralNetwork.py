import numpy as np
import pickle as pk

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
        return input##########quaaa

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