import numpy as np

from layer import Layer


class DenseLayer(Layer):
    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_forward_prop()
        return self.outputs

    def initialize_weights_and_bias(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def perform_forward_prop(self):
        if self.weights is None:
            print('Layer is missing weights')
            return
        if self.bias is None:
            print('Layer is missing a bias')
            return

        output = np.squeeze(np.dot(self.inputs, self.weights) + self.bias)
        return output
