import numpy as np

from layer import Layer


class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_flattening()

    def perform_flattening(self):
        raise NotImplementedError
