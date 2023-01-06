import numpy as np


class MaxPoolingLayer:
    def __init__(self, input_shape):
        # Layer input and its size
        self.input = None
        self.input_shape = input_shape

        # Shape of the kernel used in pooling
        self.kernel_shape = (2, 2)

        # Layer output and its size
        self.output_shape = (input_shape[0] // self.kernel_shape[0],
                             input_shape[1] // self.kernel_shape[1],
                             input_shape[2])
        self.output = np.zeros(self.output_shape)

        # Prepare delta variable for backpropagation
        self.delta = None

    def forward_prop(self, layer_input):
        self.input = layer_input

        row_range = range(0, self.input_shape[0], self.kernel_shape[0])
        col_range = range(0, self.input_shape[1], self.kernel_shape[1])

        # For each filter in layer
        for f in range(self.input_shape[2]):
            # For each row
            for r in row_range:
                r_end = r + self.kernel_shape[0]

                # For each column
                for c in col_range:
                    c_end = c + self.kernel_shape[1]

                    # Get a chunk of the input array
                    chunk = self.input[r: r_end, c: c_end, f]

                    # Determine output array indices and perform max-pooling
                    output_row = r // self.kernel_shape[0]
                    output_col = c // self.kernel_shape[1]
                    self.output[output_row, output_col, f] = np.max(chunk)

        return self.output

    def backward_prop(self, next_layer):    # TODO: check and maybe switch for the same loop as forward_prop()
        # For each filter in layer
        for f in range(self.output_shape[2]):
            # For each row
            for r in range(self.output_shape[0]):
                # For each column
                for c in range(self.output_shape[1]):
                    # Get delta from downstream
                    delta_output = next_layer.delta[r, c, f]

                    # Determine input array indices and get its chunk
                    input_row_start = r * self.kernel_shape[0]
                    input_row_end = input_row_start + self.kernel_shape[0]

                    input_col_start = c * self.kernel_shape[1]
                    input_col_end = input_col_start + self.kernel_shape[1]

                    chunk = self.input[input_row_start: input_row_end,
                                       input_col_start: input_col_end,
                                       f]
                    max_value = np.max(chunk)
                    max_value_index = np.argwhere(chunk == max_value)[0]

                    self.delta[r + max_value_index[0],
                               c + max_value_index[1],
                               f] = delta_output


