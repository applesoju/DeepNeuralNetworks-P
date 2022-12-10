import numpy as np

from layer import Layer


class DropoutLayer(Layer):
    def __init__(self):
        super().__init__()
        self.propability = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_dropout()
        return self.outputs

    def set_propability(self, prop):
        self.propability = prop

    def perform_dropout(self):
        output = np.zeros(self.inputs.shape)

        for i, _ in enumerate(output):
            if np.random.rand() < 1 - self.propability:
                output[i] = self.inputs[i]

        return output
