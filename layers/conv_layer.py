import numpy as np

from .funs import linear, linear_prime
from .im2col import im2col_indices


class ConvolutionalLayer:
    def __init__(self, input_shape=None, n_filters=1, kernel_shape=(3, 3),
                 activation=linear, activation_deriv=linear_prime):
        # Layer input and its shape
        self.input = None
        self.input_shape = input_shape if len(input_shape) == 4 else input_shape + (1, 1)

        # Number of filters used in the layer and their shape
        self.n_filters = n_filters
        self.kernel_shape = (self.n_filters,  # Number of filters
                             self.input_shape[2],  # Number of channels in images
                             kernel_shape[0],  # Kernel height
                             kernel_shape[1])  # Kernel width

        # Prepare padded input
        self.padding = self.kernel_shape[2] // 2
        # padded_input_shape = (self.input_shape[0] + 2 * self.padding,     # not needed?
        #                       self.input_shape[1] + 2 * self.padding,
        #                       self.input_shape[2])
        # self.padded_input = np.zeros(padded_input_shape)

        # Initialization of filters and biases using normal distribution
        standard_dev = 1 / np.sqrt(np.prod(self.kernel_shape))
        self.weights = np.random.normal(0, standard_dev, self.kernel_shape)
        self.biases = np.random.randn(n_filters)

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_deriv = activation_deriv

        # Layer output and its shape assuming (1, 1) strides and padding that won't cause size change
        self.output_shape = (self.input_shape[0],  # Input height
                             self.input_shape[1],  # Input width
                             n_filters,  # Number of filters (channels)
                             self.input_shape[3])  # Number of inputs
        self.output = None

        # Prepare delta variables for backpropagation
        self.delta = None
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_biases = np.zeros(self.biases.shape)

    def forward_prop(self, layer_input):
        h, w, c, n = self.input_shape

        if len(layer_input.shape) == 2:  # First layer case
            self.input = layer_input[np.newaxis, np.newaxis, :]

        else:  # Not first layer case
            self.input = layer_input

        convo_result_shape = self.n_filters, h, w, n

        kernel_h = self.kernel_shape[2]  # Kernel height
        kernel_w = self.kernel_shape[3]  # Kernel width

        # Convert images into columns
        input_col = im2col_indices(self.input, kernel_h, kernel_w, self.padding)
        weights_col = self.weights.reshape(self.n_filters, -1)

        # Perform convolution
        out = weights_col @ input_col + self.biases[:, np.newaxis]
        out = out.reshape(convo_result_shape).transpose(1, 2, 0, 3)

        # Activate outputs
        self.output = self.activation(out)

        return self.output

    def backward_prop(self, next_layer):
        self.delta = np.zeros(self.input_shape)

        # For every filter
        for f in range(self.n_filters):
            # For every row
            for r in range(self.kernel_shape[0], self.input_shape[0] + 1):
                r_start = r - self.kernel_shape[0]

                # For every column
                for c in range(self.kernel_shape[1], self.input_shape[1] + 1):
                    c_start = c - self.kernel_shape[1]

                    # Get a chunk of the input array
                    chunk = self.input[r_start: r, c_start: c]

                    # Determine delta terms for weights and biases
                    self.delta_weights[:, :, :, f] += \
                        chunk * next_layer.delta[r_start, c_start, f]
                    self.delta[r_start: r, c_start: c] += \
                        next_layer.delta[r_start, c_start, f] * self.weights[:, :, :, f]

            self.delta_biases[f] = np.sum(next_layer.delta[:, :, f])
        self.delta = self.activation_deriv(self.delta)

#   im2col
