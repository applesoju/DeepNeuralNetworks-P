import numpy as np

from layer import Layer


class ActivationLayer(Layer):
    def __init__(self):
        super().__init__()
        self.activation_function = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_activation()
        return self.outputs

    def set_activation_function(self, activ_fun):
        self.activation_function = activ_fun

    def perform_activation(self):
        if self.activation_function is None:
            print('Layer is missing an activation function')
            return

        vectorized_af = np.vectorize(self.activation_function)
        output = vectorized_af(self.inputs)

        return output
