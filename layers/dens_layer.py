import numpy as np


class DenseLayer:
    def __init__(self, input_size, n_neurons, activation, activation_prime):
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
        self.weights = np.random.rand(input_size, n_neurons) - 0.5
        self.biases = np.random.rand(1, n_neurons)

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None
        self.delta_weights = 0
        self.delta_biases = 0

    def forward_prop(self, layer_input):
        # Dot product of input and neuron weights plus bias values
        dense_output = np.dot(layer_input, self.weights) + self.biases
        # Activate output using provided function
        activated_output = self.activation(dense_output)

        return activated_output

    def backward_prop(self, next_layer):
        # Compute error from downstream and determine this layers delta term
        self.error = np.dot(next_layer.weights, next_layer.delta)
        self.delta = self.error * self.activation_prime(self.output)

        # Determine delta terms for weights and biases
        self.delta_weights += self.delta * np.atleast_2d(self.input).T
        self.delta_biases += self.delta
