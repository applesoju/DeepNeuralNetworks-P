import numpy as np

from .layer import Layer


class DropoutLayer(Layer):
    def __init__(self, probability):
        super().__init__()

        self.probability = probability
        self.mask = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_dropout()
        return self.outputs

    def perform_dropout(self):
        output = np.zeros(self.inputs.shape)

        self.mask = (np.random.rand(*output.shape) > self.probability) / self.probability
        output = self.inputs * self.mask

        return output

    def perform_backward_prop(self, output_err, learn_rate):
        return output_err * self.mask
