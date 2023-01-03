import numpy as np


class DenseLayer:
    def __init__(self, input_shape, n_neurons, activation, activation_prime, weights=None, bias=None):
        self.input_shape = input_shape
        self.input = None

        self.n_neurons = n_neurons
        self.activation = activation
        self.activation_prime = activation_prime

        self.weights = np.random.rand(input_size, n_neurons) - 0.5 if weights is not None else weights
        self.bias = np.random.rand(1, n_neurons) - 0.5 if bias is not None else bias

    def forward_prop(self):
        raise NotImplementedError

    def backward_prop(self):
        raise NotImplementedError