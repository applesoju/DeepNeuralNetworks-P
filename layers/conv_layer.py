import numpy as np

from .funs import linear, linear_prime
from .im2col import im2col_indices, col2im_indices


class ConvolutionalLayer:
    def __init__(self, input_shape=None, n_filters=1, kernel_shape=(3, 3),
                 activation=linear, activation_deriv=linear_prime):
        # Layer input and its shape
        self.input = None
        self.input_col = None
        self.input_shape = None

        # Number of filters used in the layer and their shape
        self.n_filters = n_filters
        self.filter_shape = kernel_shape
        self.kernel_shape = None

        self.padding = None

        self.weights = None
        self.biases = None

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_deriv = activation_deriv

        self.output_shape = None
        self.output = None

        self.delta = None
        self.col_delta_weights = None
        self.delta_weights = None
        self.delta_biases = None

        self.name = 'Convolutional Layer'

        if input_shape is not None:
            self.init_params(input_shape)

    def init_params(self, input_shape):
        if len(input_shape) == 4:
            self.input_shape = input_shape

        elif len(input_shape) == 3:
            self.input_shape = input_shape + (1, )

        elif len(input_shape) == 2:
            self.input_shape = input_shape + (1, 1)

        else:
            raise ValueError('ConvolutionalLayer input must be of dimensions 2, 3 or 4.')

        self.kernel_shape = (self.n_filters,  # Number of filters
                             self.input_shape[2],  # Number of channels in images
                             self.filter_shape[0],  # Kernel height
                             self.filter_shape[1])  # Kernel width

        # Prepare padded input
        self.padding = self.kernel_shape[2] // 2

        # Initialization of filters and biases using normal distribution
        standard_dev = 1 / np.sqrt(np.prod(self.kernel_shape))
        self.weights = np.random.normal(0, standard_dev, self.kernel_shape)
        self.biases = np.random.randn(self.n_filters)

        # Layer output and its shape assuming (1, 1) strides and padding that won't cause size change
        self.output_shape = (self.input_shape[0],  # Input height
                             self.input_shape[1],  # Input width
                             self.n_filters,  # Number of filters (channels)
                             self.input_shape[3])  # Number of inputs

        # Prepare delta variables for backpropagation
        self.col_delta_weights = np.zeros((self.n_filters,
                                           self.input_shape[2] * self.kernel_shape[2] * self.kernel_shape[3]))
        self.delta_weights = np.zeros(self.weights.shape)
        self.delta_biases = np.zeros(self.biases.shape)

    def forward_prop(self, layer_input):
        h, w, c, n = self.input_shape

        if len(layer_input.shape) == 2:  # First layer case
            self.input = layer_input[:, :, np.newaxis, np.newaxis]

        else:  # Not first layer case
            self.input = layer_input.reshape(self.input_shape)

        input_reshaped = self.input.transpose(3, 2, 0, 1)
        convo_result_shape = self.n_filters, h, w, n

        kernel_h = self.kernel_shape[2]  # Kernel height
        kernel_w = self.kernel_shape[3]  # Kernel width

        # Convert images into columns
        self.input_col = im2col_indices(input_reshaped, kernel_h, kernel_w, self.padding)
        weights_col = self.weights.reshape(self.n_filters, -1)

        # Perform convolution
        out = weights_col @ self.input_col + self.biases[:, np.newaxis]
        out = out.reshape(convo_result_shape).transpose(1, 2, 0, 3)

        # Activate outputs
        self.output = self.activation(out)

        return self.output

    def backward_prop(self, next_layer):
        if len(next_layer.delta.shape) == 3:
            delta_nl = next_layer.delta[:, :, :, np.newaxis]

        else:
            delta_nl = next_layer.delta

        kernel_h = self.kernel_shape[2]  # Kernel height
        kernel_w = self.kernel_shape[3]  # Kernel width

        # Get delta term from next layer in needed shapes
        delta_nxt_layer = delta_nl.transpose(3, 2, 0, 1)
        delta_result_shaped = delta_nxt_layer.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        # Determine delta term of biases
        self.delta_biases = np.sum(delta_nxt_layer, axis=(0, 2, 3))

        # Determine delta term of weights
        self.col_delta_weights += delta_result_shaped @ self.input_col.T
        self.delta_weights += self.col_delta_weights.reshape(self.weights.shape)

        # Get weights in needed shape
        weights_reshaped = self.weights.reshape(self.n_filters, -1)

        # Determine delta term of inputs
        delta_col = weights_reshaped.T @ delta_result_shaped
        self.delta = col2im_indices(delta_col, self.input_shape, kernel_h, kernel_w, padding=self.padding)
        self.delta = self.activation_deriv(self.delta)
