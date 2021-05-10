import numpy as np


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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2
    
def sum_square_error(predicted, target):
    return np.sum((predicted - target)**2) / 2
def sum_square_error_derivative(predicted, target):
    return predicted - target