import numpy as np


class DenseLayer:
    def __init__(self, n_neurons, activation, activation_deriv, input_shape=None):
        # Layer input and its shape
        self.input = None
        self.input_shape = None

        # Layer output and number of neurons
        self.output = None
        self.output_shape = None
        self.n_neurons = n_neurons

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_deriv = activation_deriv

        # Weights and biases of the layer
        self.weights = None
        self.biases = None

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None
        self.delta_weights = None
        self.delta_biases = None

        self.name = 'Dense Layer'

        if input_shape is not None:
            self.init_params(input_shape)

    def init_params(self, input_shape):
        self.input_shape = input_shape[1]

        self.output_shape = (1, self.n_neurons)

        # Weights and biases of the layer
        self.weights = np.random.rand(self.input_shape, self.n_neurons) - 0.5
        self.biases = np.random.rand(1, self.n_neurons)

        # Prepare delta variables for backpropagation
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_biases = np.zeros(self.biases.shape)

    def forward_prop(self, layer_input):
        self.input = layer_input

        # Dot product of input and neuron weights plus bias values
        dense_output = np.dot(layer_input, self.weights) + self.biases
        # Activate output using provided function
        self.output = self.activation(dense_output)

        return self.output

    def backward_prop(self, next_layer):
        # If the next layer is Dropout just get the error
        if type(next_layer).__name__ == 'DropoutLayer':
            self.error = next_layer.error

        # If not compute error from downstream
        else:
            self.error = np.dot(next_layer.weights, next_layer.delta.T).T

        # Determine this layers delta term
        self.delta = self.error * self.activation_deriv(self.output)

        # Determine delta terms for weights and biases
        self.delta_weights += self.delta * self.input.T
        self.delta_biases += self.delta
