import numpy as np


class FlatteningLayer:
    def __init__(self, input_shape):
        # Layer input and its shape
        self.input = None
        self.input_shape = input_shape

        # Layer output
        self.output = None

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None

    def forward_prop(self, layer_input):
        self.input = layer_input
        self.output = self.input.flatten().reshape(1, -1)

        return self.output

    def backward_prop(self, next_layer):
        # Compute error from downstream and determine this layers delta term
        self.error = np.dot(next_layer.weights, next_layer.delta.T).T
        self.delta = self.error * self.output
        self.delta = self.delta.reshape(self.input_shape)
