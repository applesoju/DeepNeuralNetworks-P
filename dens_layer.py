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

    def perform_forward_prop(self):
        output = np.dot(self.inputs, self.weights) + self.bias
        return output
