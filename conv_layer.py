import numpy as np

from layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self):
        super().__init__()
        self.activation_function = None
        self.filter = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        convoluted_output = self.perform_convolution(1)
        print(convoluted_output)

    def set_filter_and_activation_fun(self, filter_obj, act_fun):  # TODO: more than one filter
        self.filter = filter_obj
        self.activation_function = act_fun

    def perform_convolution(self, padding=0, strides=1):
        input_shape = (self.inputs.shape[0], self.inputs.shape[1])

        if self.activation_function is None or self.filter is None:
            print('Layer is missing a filter and/or an activation function.')
            return

        kernel = np.flipud(np.fliplr(self.filter))
        output_shape = (int(((input_shape[0] - kernel.shape[0] + 2 * padding) / strides) + 1),
                        int(((input_shape[1] - kernel.shape[1] + 2 * padding) / strides) + 1))

        output = np.zeros(output_shape)

        if padding == 0:
            output_padded = self.inputs

        else:
            output_padded = np.zeros((input_shape[0] + padding * 2,
                                      input_shape[1] + padding * 2))

            output_padded[int(padding): int(padding * -1),
                          int(padding): int(padding * -1)] = self.inputs

        for row in range(input_shape[1]):
            # if row > input_shape[1] - kernel.shape[1]:
            #     break

            if row % strides != 0:
                continue

            for col in range(input_shape[0]):
                # if col > input_shape[0] - kernel.shape[0]:
                #     break

                if col % strides == 0:
                    output[col, row] = (
                        kernel * output_padded[col: col + kernel.shape[0],
                                               row: row + kernel.shape[1]]
                    ).sum()

        self.outputs = output
        return self.outputs
