import numpy as np

from .funs import linear, linear_prime
from .timer import Timer
from .im2col import im2col_indices


class ConvolutionalLayer:
    def __init__(self, input_shape=None, n_filters=1, kernel_shape=(3, 3),
                 activation=linear, activation_deriv=linear_prime):
        # Layer input and its shape
        self.input = None
        self.input_shape = input_shape if len(input_shape) == 3 else input_shape + (1, )

        # Number of filters used in the layer and their shape
        self.n_filters = n_filters
        self.kernel_shape = (kernel_shape[0], kernel_shape[1], self.input_shape[2], self.n_filters)

        # Prepare padded input
        self.padding = self.kernel_shape[0] // 2
        padded_input_shape = (self.input_shape[0] + 2 * self.padding,
                              self.input_shape[1] + 2 * self.padding,
                              self.input_shape[2])
        self.padded_input = np.zeros(padded_input_shape)

        # Initialization of filters and biases using normal distribution
        standard_dev = 1 / np.sqrt(np.prod(self.kernel_shape))
        self.weights = np.random.normal(0, standard_dev, self.kernel_shape)
        self.biases = np.random.randn(n_filters)

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_deriv = activation_deriv

        # Layer output and its shape assuming zero padding and (1, 1) strides
        self.output_shape = (self.input_shape[0], self.input_shape[1], n_filters)
        self.output = np.zeros(self.output_shape)

        # Prepare delta variables for backpropagation
        self.delta = None
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_biases = np.zeros(self.biases.shape)

    def forward_prop(self, layer_input):
        self.input = layer_input.reshape(self.input_shape)

        # Apply zero padding
        self.padded_input[self.padding: -self.padding, self.padding: -self.padding, :, :] = self.input
        all_f_timer = Timer()
        # For each filter in layer
        for f in range(self.n_filters):
            # For each row
            # filter_timer = Timer()
            for r in range(self.input_shape[0]):
                r_end = r + self.kernel_shape[0]

                # For each column
                for c in range(self.input_shape[1]):
                    c_end = c + self.kernel_shape[1]

                    # Get a chunk of the padded input array
                    chunk = self.padded_input[r: r_end, c: c_end]

                    # Perform convolution
                    convolution_output = (chunk * self.weights[:, :, :, f]).sum()  # + self.biases[f]
                    # convolution_output = np.multiply(chunk, self.weights[:, :, :, f]).sum() + self.biases[f]
                    self.output[r, c, f] = convolution_output
            # print('One filter convolution time:')
            # filter_timer.stop(True)
        print('All filters timer:')
        all_f_timer.stop(True)

        # Activate outputs
        self.output = self.activation(self.output)

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
                    self.delta_weights[:, :, :, f] +=\
                        chunk * next_layer.delta[r_start, c_start, f]
                    self.delta[r_start: r, c_start: c] +=\
                        next_layer.delta[r_start, c_start, f] * self.weights[:, :, :, f]

            self.delta_biases[f] = np.sum(next_layer.delta[:, :, f])
        self.delta = self.activation_deriv(self.delta)

#   im2col
