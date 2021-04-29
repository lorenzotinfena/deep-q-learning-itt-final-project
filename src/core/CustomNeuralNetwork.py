import numpy as np
import pickle as pk
from core.NeuralNetwork import NeuralNetwork


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
            x = np.copy(x)
            x[x<0] = 0
            return x
        def ReLU_derivative(x):
            x = np.copy(x)
            x[x<=0] = 0
            x[x>0] = 1
            return x

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
        self.activation_functions = [identity]*3
        self.activation_functions_derivative = [identity_derivative]*3
        self.cost_function = sum_square_error
        self.cost_function_derivative = sum_square_error_derivative