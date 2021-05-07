import numpy as np
import pickle as pk
from core.neural_network import NeuralNetwork

class CartPoleNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__([4, 10, 2])

        def identity(x): return x
        def identity_derivative(x): return np.ones(len(x))

        def ReLU(x):
            return np.multiply(x,(x>0))
        def ReLU_derivative(x):
            return (x>0)*1

        def leaky_relu(x):
            return np.maximum(0.1*x, x)
        def leaky_relu_derivative(x):
            res = np.ones_like(x)
            res[x < 0] = 0.1
            return res
        
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
        self._activation_functions = [leaky_relu, identity]
        self._activation_functions_derivative = [leaky_relu_derivative, identity_derivative]
        self.cost_function = sum_square_error
        self._cost_function_derivative = sum_square_error_derivative