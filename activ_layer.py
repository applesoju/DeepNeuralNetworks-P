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

        output = np.zeros(self.inputs.shape)
        print(output)

        for i, row in enumerate(self.inputs):
            for j, elem in enumerate(row):
                output[i, j] = self.activation_function(elem)

        return output

