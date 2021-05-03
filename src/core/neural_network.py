import numpy as np
import pickle as pk


class NeuralNetwork:
    def __init__(self, n_neurons: np.array):
        """
        args:
            n_neurons: np.array
                number of neurons per layer
			path: path to weights and biases dumped
		"""
        self._n_neurons = n_neurons
        self.weights = [np.random.uniform(low=-0.5, high=0.5, size=(n_neurons[i+1], n_neurons[i]+1)) for i in range(len(n_neurons) - 1)]

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
		Args:
			input: np.ndarray 1dim
                array that rapresent inputs
		Return:
			prediction: np.ndarray 1dim
		"""
        for weights, activation_function in zip(self.weights, self._activation_functions):
            input = activation_function(weights.dot(np.append(input, 1)))
        return input
    
    def save(self, path: str):
        with open(path, "wb") as file:
            pk.dump(self.weights, file)

    def load(self, path: str):
        with open(path, 'rb') as file:
                self.weights = pk.load(file)

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
        for weights, activation_function in zip(self.weights, self._activation_functions):
            z.append(weights.dot(np.append(a[-1], 1)))
            a.append(activation_function(z[-1]))
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
        gradients_a = self._cost_function_derivative(a[-1], target_output)
        gradients_z = gradients_a * self._activation_functions_derivative[-1](z[-1])
        
        # update weights and biases, and compute gradients for hidden layers
        for weights, z_layer, a_layer, activation_function_derivative in reversed(list(zip(self.weights[1:], z[1:-1], a[1:-1], self._activation_functions_derivative[:-1]))):
            weights -= learning_rate * gradients_z.reshape(-1, 1).dot(np.append(a_layer, 1).reshape(1, -1))
            gradients_a = weights.T.dot(gradients_z)[:-1]
            gradients_z = gradients_a * activation_function_derivative(z_layer)

        #update first weights and biases
        self.weights[0] -= learning_rate * gradients_z.reshape(-1, 1).dot(np.append(a[0], 1).reshape(1, -1))
        
        # check weights integrity
        if any([any([any(np.isnan(weights)) for weights in weights]) for weights in self.weights]):
            raise Exception('NaN weights')