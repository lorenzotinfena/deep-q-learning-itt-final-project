import numpy as np
import pickle as pk

class NeuralNetwork:
    def __init__(self, n_neurons: np.ndarray, activation_functions: np.array, cost_function_derivative, seed=None):
        """
		Args:
			n_neurons: np.ndarray[layer, j]
                number of neurons per layer
            activation_functions: np.array dtype:function(np.array, derivative=False)->np.array
            cost_function_derivative: function(np.array)->np.array
            seed: int = None
                seed to use when initializing weights
		"""
        np.random.seed(seed)
        self.activation_functions = activation_functions
        self.cost_function_derivative = cost_function_derivative
        self.bias = [(np.random.randn(layer)) for layer in s[1:]]
        self.s = s
        self.weights = [(np.random.randn(n_neurons[i + 1], n_neurons[i])) for i in range(len(n_neurons) - 1)]

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
		Args:
			input: np.ndarray 1dim
                array that rapresent inputs
		Return:
			prediction: np.ndarray 1dim
		"""
        for i, thetas, bias in enumerate(zip(self.thetas, self.bias)):
            input = self.activation_functions[i](thetas.dot(input) + bias)
        return input.T[0]

    def forward_propagation(self, input: np.ndarray) -> np.ndarray,  np.ndarray:
        '''
        Args:
            input: np.ndarray 1dim
                array that rapresent inputs
        Return: predict values
            a, z
        '''
        a = [input]
        z = [input]
        for thetas, bias in zip(self.thetas, self.bias):
            z_ = thetas.dot(input) + bias
            z.append(z_)
            input = self.activation_functions[i](z_)
            a.append(input)
        return np.array(a), np.array(z)

    def backpropagation(self, a: np.ndarray, z: np.ndarray, learning_rate: float):
        '''
        Args:
            a: np.ndarray
            z: np.ndarray
            learning_rate: float
        '''
        error_supp = self.Y - a[-1]
        delta_supp = error_supp * self.__sigmoid_derivate(z[-1])
        for thetas, bias, a, z in enumeratezip(reversed(self.thetas), reversed(self.bias), reversed(a[:-1]), reversed(z[:-1])):
            bias += learning_rate * np.sum(delta_supp, axis=0)
            thetas += learning_rate * delta_supp.T.dot(a)
            error_supp = delta_supp.dot(thetas)
            delta_supp = error_supp * self.__sigmoid_derivate(z)

    def cost_function(self) -> float:
        y = self.Y.reshape(-1)
        x = self.multi_predict(self.X).reshape(-1)
        return np.sum((x - y) ** 2) / self.Y.shape[0] * self.Y.shape[1]

    def fit(self, X: np.ndarray, Y: np.ndarray learning_rate: float):
        if self.s[0] != X.shape[1] or self.s[-1] != Y.shape[1]:
            raise Exception('s parameter is incorrect')
        if X.shape[0] != Y.shape[0]:
            raise Exception('X or Y first shape aren t same')
        
        diff = int(n_iterations / 10) + 1
        print('Errore quadratico medio')
        for i in range(n_iterations):
            self.__backpropagation(learning_rate)
            if i % diff == 0:
                print(self.cost_function())
    def __sigmoid(self, x: float) -> float:
        res = np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
        return res

    def __sigmoid_derivate(self, x: float) -> float:
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    def save(self, path):
        with open(path, "wb") as file:
            pk.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            newobj = pk.load(file)
            return newobj

def main():
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    Y = np.array(([40], [86], [89]), dtype=float)
    X = X / np.amax(X, axis=0)  # maximum of X array
    Y = Y / 100  # maximum test score is 100
    NN = NeuralNetwork(X, Y, np.array([2, 3, 1]))
    NN.fit(1000, 1)
    print()
    print('final cost is:')
    print(NN.cost_function())
    print()
    print("inputs are:")
    print(NN.X)
    print()
    print("expected outputs are:")
    print(NN.Y)
    print()
    print("Predicted outputsassssssss are:")
    print(NN.multi_predict(NN.X).T)

if __name__ == '__main__':
    main()