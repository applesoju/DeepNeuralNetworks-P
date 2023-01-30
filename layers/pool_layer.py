import numpy as np
from .im2col import im2col_indices, col2im_indices


class MaxPoolingLayer:
    def __init__(self, input_shape=None):
        self.input = None
        self.input_cols = None
        self.input_shape = None

        # Shape of the kernel used in pooling
        self.kernel_shape = (2, 2)

        self.output_shape = None
        self.output = None

        self.delta = None

        self.name = 'MaxPooling Layer'

        if input_shape is not None:
            self.init_params(input_shape)

    def init_params(self, input_shape):
        self.input_shape = input_shape

        if len(input_shape) == 2:
            self.input_shape = self.input_shape + (1, 1)

        # Layer output and its size
        self.output_shape = (self.input_shape[0] // self.kernel_shape[0],
                             self.input_shape[1] // self.kernel_shape[1],
                             self.input_shape[2],
                             self.input_shape[3])
        self.output = np.zeros(self.output_shape)

        # Prepare delta variable for backpropagation
        self.delta = np.zeros(self.input_shape)

    def forward_prop(self, layer_input):
        self.input = layer_input.reshape(self.input_shape)

        input_reshaped = self.input.transpose(3, 2, 0, 1)

        # Get input dimensions (shape)
        h, w, c, n = self.input_shape

        # Get shape of pooling kernel
        h_pool = self.kernel_shape[0]
        w_pool = self.kernel_shape[1]

        # Determine output shape
        h_out = (h - h_pool) // h_pool + 1
        w_out = (w - w_pool) // w_pool + 1

        input_split = input_reshaped.reshape(n * c, 1, h, w)
        self.input_cols = im2col_indices(input_split, h_pool, w_pool, padding=0, stride=h_pool)

        # Get indices of max values and save those values
        input_cols_argmax = np.argmax(self.input_cols, axis=0, keepdims=True)
        input_cols_max = self.input_cols[input_cols_argmax, np.arange(input_cols_argmax.shape[1])]

        # Get a proper shape
        self.output = input_cols_max.reshape(h_out, w_out, n, c).transpose(0, 1, 3, 2)

        return self.output

    def backward_prop(self, next_layer):
        delta_nl_reshaped = next_layer.delta.transpose(3, 2, 0, 1)

        # Get input dimensions (shape)
        h, w, c, n = self.input_shape

        # Get shape of pooling kernel
        h_pool = self.kernel_shape[0]
        w_pool = self.kernel_shape[1]

        delta_nl_trans = delta_nl_reshaped.transpose(2, 3, 0, 1).flatten()
        delta_cols = np.zeros_like(self.input_cols)

        input_cols_argmax = np.argmax(self.input_cols, axis=0, keepdims=True)
        delta_cols[input_cols_argmax, np.arange(input_cols_argmax.shape[1])] = delta_nl_trans

        # Determine delta term for this layer
        delta = col2im_indices(delta_cols, (n * c, 1, h, w), h_pool, w_pool, padding=0, stride=h_pool)

        input_reshaped = self.input.transpose(3, 2, 0, 1)
        self.delta = delta.reshape(input_reshaped.shape).transpose(2, 3, 1, 0)

    # Source: https://github.com/yunjey/cs231n/
