import numpy as np


class DenseLayer:
    def __init__(self, input_size, n_neurons, activation, activation_prime, weights=None, bias=None):
        # Layer input and its size
        self.input = None
        self.input_size = input_size

        # Layer output and the number of neurons in the layer (size of the output)
        self.output = None
        self.n_neurons = n_neurons

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_prime = activation_prime

        # Weights and biases of the layer
        self.weights = np.random.rand(input_size, n_neurons) - 0.5 if weights is not None else weights
        self.biases = np.random.rand(1, n_neurons) - 0.5 if bias is not None else bias

    def forward_prop(self, layer_input):
        # Dot product of input and neuron weights plus bias values
        dense_output = np.dot(layer_input, self.weights) + self.biases
        # Activate output using provided function
        activated_output = self.activation(dense_output)

        return activated_output

    def backward_prop(self, next_layer):
        raise NotImplementedError