import numpy as np

from layer import Layer


class PoolingLayer(Layer):
    def __init__(self):
        super().__init__()
        self.kernel_shape = None
        self.n_strides = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_pooling()
        return self.outputs

    def set_kernel_and_strides(self, kernel_shape, n_strides):
        self.kernel_shape = kernel_shape
        self.n_strides = n_strides

    def perform_pooling(self, method='max'):
        input_shape = (self.inputs.shape[0], self.inputs.shape[1])

        output_shape = (
            int(1 + (input_shape[0] - self.kernel_shape[0]) / self.n_strides),
            int(1 + (input_shape[1] - self.kernel_shape[1]) / self.n_strides)
        )

        output = np.zeros(output_shape)

        for x in range(output_shape[0]):
            for y in range(output_shape[1]):
                x_start = x * self.n_strides
                x_end = x_start + self.kernel_shape[0]

                y_start = y * self.n_strides
                y_end = y_start + self.kernel_shape[1]

                pool_slice = self.inputs[x_start: x_end, y_start: y_end]

                match method:
                    case 'max':
                        output[x, y] = np.max(pool_slice)
                    case 'avg':
                        output[x, y] = np.average(pool_slice)

        return output
