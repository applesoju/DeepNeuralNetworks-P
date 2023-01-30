import numpy as np


class FlatteningLayer:
    def __init__(self, input_shape=None):
        # Layer input and its shape
        self.input = None
        self.input_shape = None

        # Layer output and its shape
        self.output = None
        self.output_shape = None
        self.n_neurons = None

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None

        self.name = 'Flattening Layer'

        if input_shape is not None:
            self.init_params(input_shape)

    def init_params(self, input_shape):
        self.input_shape = input_shape

        # Get output shape from shape of input
        output_shape = 1
        for i in self.input_shape:
            output_shape *= i

        self.n_neurons = output_shape
        self.output_shape = (1, output_shape)

    def forward_prop(self, layer_input):
        # Perform flattening
        self.input = layer_input
        self.output = self.input.flatten().reshape(1, -1)

        return self.output

    def backward_prop(self, next_layer):
        # Compute error from downstream and determine this layers delta term
        self.error = np.dot(next_layer.weights, next_layer.delta.T).T
        self.delta = self.error * self.output
        self.delta = self.delta.reshape(self.input_shape)
