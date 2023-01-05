import numpy as np
from .funs import linear, linear_prime


class ConvolutionalLayer:
    def __init__(self, input_shape=None, n_filters=1, kernel_shape=(3, 3),
                 activation=linear, activation_deriv=linear_prime):
        # Layer input and its shape
        self.input = None
        self.input_shape = input_shape

        # Number of filters used in the layer and their shape
        self.n_filters = n_filters
        self.kernel_shape = kernel_shape if input_shape is None else kernel_shape + (n_filters, )

        # Initialization of filters and biases using normal distribution
        standard_dev = 1 / np.sqrt(np.prod(self.kernel_shape))
        self.weights = np.random.normal(0, standard_dev, self.kernel_shape)
        self.biases = np.random.randn(n_filters)

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_deriv = activation_deriv

        # Layer output and its shape assuming zero padding and (1, 1) strides
        self.output_shape = self.input_shape + (n_filters, )
        self.output = np.zeros(self.output_shape)

        # Prepare delta variables for backpropagation
        self.delta = np.zeros(self.input_shape + (n_filters, ))
        self.delta_weights = 0
        self.delta_biases = 0

    def forward_prop(self, layer_input):
        self.input = layer_input

        # Apply zero padding
        padding = self.kernel_shape[0] // 2
        padded_input_shape = self.input_shape[0] + 2 * padding, self.input_shape[1] + 2 * padding

        padded_input = np.zeros(padded_input_shape)
        padded_input[padding: -padding, padding: -padding] = layer_input

        output = np.zeros(self.output_shape)

        # For every filter in layer
        for f in range(self.n_filters):
            # For every row
            for r in range(self.input_shape[0]):
                r_end = r + self.kernel_shape[0]

                # For every column
                for c in range(self.input_shape[1]):
                    c_end = c + self.kernel_shape[1]

                    # Get a chunk of the padded input array
                    chunk = padded_input[r: r_end, c: c_end]

                    # Perform convolution
                    convolution_output = (chunk * self.weights[:, : , f]).sum()  + self.biases[f]
                    output[r, c, f] = convolution_output

        # Activate outputs
        self.output = self.activation(output)

        return self.output


    def backward_prop(self, next_layer):    # TODO
        for f in range(self.n_filters):
            # For every row
            for r in range(self.kernel_shape[0], self.input_shape[0]):  # TODO: fix like in forward
                r_start = r - self.kernel_shape[0]

                # For every column
                for c in range(self.kernel_shape[1], self.input_shape[1]):  # TODO: fix like in forward
                    c_start = c - self.kernel_shape[1]

                    # Get a chunk of the input array
                    chunk = self.input[r_start: r, c_start: c]  # TODO: fix like in forward

                    # Determine delta terms for weights and biases
                    self.delta_weights[:, :, f] += chunk * next_layer.delta[r_start, c_start, f]
                    self.delta[r_start, c_start, f] += next_layer.delta[r_start, c_start, f] * self.weights[:, :, f]

            self.delta_biases[f] = np.sum(next_layer.delta[:, :, f])
        self.delta = self.activation_deriv(self.delta)