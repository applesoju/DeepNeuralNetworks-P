import numpy as np


class DenseLayer:
    def __init__(self, input_size, n_neurons, activation, activation_prime, weights=None, bias=None):
        self.input_size = input_size
        self.input = None

        self.n_neurons = n_neurons
        self.output = None

        self.activation = activation
        self.activation_prime = activation_prime

        self.weights = np.random.rand(input_size, n_neurons) - 0.5 if weights is not None else weights
        self.biases = np.random.rand(1, n_neurons) - 0.5 if bias is not None else bias

    def forward_prop(self, layer_input):
        dense_output = np.dot(layer_input, self.weights) + self.biases
        activated_output = self.activation(dense_output)

        return activated_output

    def backward_prop(self, next_layer):
        raise NotImplementedError