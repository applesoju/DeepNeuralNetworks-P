import numpy as np

from .layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self, kernel, strides=1):
        super().__init__()

        self.kernel = kernel
        self.padding = kernel.shape[0] // 2
        self.strides = strides

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_convolution()
        return self.outputs

    def perform_convolution(self):
        input_shape = (self.inputs.shape[0], self.inputs.shape[1])
        padd = self.padding
        strides = self.strides

        kernel = np.flipud(np.fliplr(self.kernel))
        output_shape = (
            int(((input_shape[0] - kernel.shape[0] + 2 * padd) / strides) + 1),
            int(((input_shape[1] - kernel.shape[1] + 2 * padd) / strides) + 1)
        )

        output = np.zeros(output_shape)

        if padd == 0:
            output_padded = self.inputs

        else:
            output_padded = np.zeros((input_shape[0] + padd * 2,
                                      input_shape[1] + padd * 2))

            output_padded[int(padd): int(padd * -1), int(padd): int(padd * -1)] = self.inputs

        for row in range(input_shape[1]):
            if row % strides != 0:
                continue

            for col in range(input_shape[0]):
                if col % strides == 0:
                    output[col, row] = (
                            kernel * output_padded[col: col + kernel.shape[0], row: row + kernel.shape[1]]
                    ).sum()

        return output

    def perform_backward_prop(self, output_err, learn_rate):  # TODO
        raise NotImplementedError
