import pickle as pk
from core.neural_network import NeuralNetwork
from functions import *


class CartPoleNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__([4, 30, 30, 2])
        """
        Define in object functions:
            activation_functions: iter[len(n_neurons)-1] dtype:function(np.array)->np.array
            activation_functions_derivative: iter[len(n_neurons)-1] dtype:function(np.array)->np.array
            cost_function_derivative: function(predicted: np.array, target: np.array)->np.array
                partial derivatives of cost_function respect to predicted values
		"""
        self._activation_functions = [ReLU, ReLU, identity]
        self._activation_functions_derivative = [ReLU_derivative, ReLU_derivative, identity_derivative]
        self.cost_function = sum_square_error
        self._cost_function_derivative = sum_square_error_derivative