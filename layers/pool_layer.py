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
        self.delta = np.zeros(self.input_shape)

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

    def backward_prop(self, next_layer):
        row_range = range(0, self.input_shape[0], self.kernel_shape[0])
        col_range = range(0, self.input_shape[1], self.kernel_shape[1])

        # For each filter in layer
        for f in range(self.output_shape[2]):
            # For each row
            for r in row_range:
                r_end = r + self.kernel_shape[0]

                # For each column
                for c in col_range:
                    c_end = c + self.kernel_shape[1]

                    # Get delta from downstream
                    delta_output = next_layer.delta[r // self.kernel_shape[0],
                                                    c // self.kernel_shape[1], f]

                    # Get a chunk of the input array and locate his max values
                    chunk = self.input[r: r_end, c: c_end, f]
                    max_value = np.max(chunk)
                    max_value_indices = np.argwhere(chunk == max_value)

                    # Save delta values on the same positions as max values
                    for indices in max_value_indices:
                        self.delta[r + indices[0], c + indices[1], f] = delta_output
