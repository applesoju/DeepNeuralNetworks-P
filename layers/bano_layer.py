import numpy as np

from .layer import Layer

class BatchNormalizationLayer(Layer):   # TODO
    def __init__(self):
        super().__init__()

    def compute_output(self, input_data):
        raise NotImplementedError

    def perform_forward_propagation(self):
        raise NotImplementedError

    def perform_backward_prop(self, output_err, learn_rate):
        raise NotImplementedError