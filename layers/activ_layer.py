import numpy as np

from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activ_fun):
        super().__init__()

        self.activation_function = activ_fun

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_activation()
        return self.outputs

    def perform_activation(self):
        vectorized_af = np.vectorize(self.activation_function)
        output = vectorized_af(self.inputs)

        return output
