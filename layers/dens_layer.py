import numpy as np

from .layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def set_input_size(self, input_size):
        self.input_size = input_size

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_forward_prop()
        return self.outputs

    def perform_forward_prop(self):
        output = np.squeeze(np.dot(self.inputs, self.weights) + self.bias)
        return output
