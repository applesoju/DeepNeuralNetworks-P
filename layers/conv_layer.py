import numpy as np

from .layer import Layer


def add_padding(arr, padding):
    if padding == 0:
        padded = arr

    else:
        padded = np.zeros((arr.shape[0] + padding * 2,
                           arr.shape[1] + padding * 2))

        padded[int(padding): int(padding * -1), int(padding): int(padding * -1)] = arr

    return padded


class ConvolutionalLayer(Layer):
    def __init__(self, kernel, strides=(1, 1)):
        super().__init__()

        self.kernel = kernel
        self.padding = kernel.shape[0] // 2
        self.strides = strides
        self.bias = np.random.rand() - 0.5

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
            int(((input_shape[0] - kernel.shape[0] + 2 * padd) / strides[0]) + 1),
            int(((input_shape[1] - kernel.shape[1] + 2 * padd) / strides[1]) + 1)
        )

        output = np.zeros(output_shape)

        input_padded = add_padding(self.inputs, padd)

        for row in range(0, input_shape[1], strides[1]):
            for col in range(0, input_shape[0], strides[0]):

                output[col, row] = (
                        kernel * input_padded[col: col + kernel.shape[0], row: row + kernel.shape[1]]
                ).sum() + self.bias

        return output

    def perform_backward_prop(self, output_err, learn_rate):  # TODO
        input_shape = self.inputs.shape
        kernel_shape = self.kernel.shape
        padd = self.padding
        strides = self.strides

        input_padded = add_padding(self.inputs, padd)

        input_err = np.zeros(input_padded.shape)
        kernel_err = np.zeros(kernel_shape)
        bias_err = np.sum(output_err)

        for row in range(0, input_shape[1], strides[1]):
            row_end = row + kernel_shape[1]
            row_skip = row // strides[1]

            for col in range(0, input_shape[0], strides[0]):
                col_end = col + kernel_shape[0]
                col_skip = col // strides[0]

                input_err[col: col_end, row: row_end] += \
                    output_err[col_skip, row_skip] * self.kernel

        input_err = input_err[padd: -padd, padd: -padd]

        for row in range(output_err.shape[1]):
            row_start = row * strides[1]
            row_end = row_start + kernel_shape[1]

            for col in range(output_err.shape[0]):
                col_start = col * strides[0]
                col_end = col_start + kernel_shape[0]

                kernel_err += output_err[col, row] * input_padded[col_start: col_end, row_start: row_end]

        self.kernel -= kernel_err * learn_rate
        self.bias -= bias_err * learn_rate

        return input_err