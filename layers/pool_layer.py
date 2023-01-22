import numpy as np


class MaxPoolingLayer:
    def __init__(self, input_shape=None):
        self.input = None
        self.input_reshaped = None
        self.input_shape = None

        # Shape of the kernel used in pooling
        self.kernel_shape = (2, 2)

        self.output_shape = None
        self.output = None

        self.delta = None

        if input_shape is not None:
            self.init_params(input_shape)

    def init_params(self, input_shape):
        self.input_shape = input_shape

        # Layer output and its size
        self.output_shape = (input_shape[0] // self.kernel_shape[0],
                             input_shape[1] // self.kernel_shape[1],
                             input_shape[2])
        self.output = np.zeros(self.output_shape)

        # Prepare delta variable for backpropagation
        self.delta = np.zeros(self.input_shape)

    def forward_prop(self, layer_input):
        self.input = layer_input

        h, w, c, n = self.input_shape
        kernel_h, kernel_w = self.kernel_shape

        # Reshape input to prepare it for pooling
        self.input_reshaped = layer_input.transpose(3, 2, 0, 1).reshape(n, c,
                                                                        h // kernel_h, kernel_h,
                                                                        w // kernel_w, kernel_w)

        # Perform maxpooling
        out = self.input_reshaped.max(axis=3).max(axis=4)
        self.output = out.transpose(2, 3, 1, 0)

        return self.output

    def backward_prop(self, next_layer):
        # Reshape and prepare needed arrays
        delta_nl_reshaped = next_layer.delta[:, :, :, np.newaxis].transpose(3, 2, 0, 1)
        reshaped_output = self.output.transpose(3, 2, 0, 1)
        reshaped_delta = np.zeros_like(self.input_reshaped)

        # Reshape output and make a mask by comparing it to the input
        out_newaxis = reshaped_output[:, :, :, np.newaxis, :, np.newaxis]
        mask = (self.input_reshaped == out_newaxis)

        delta_nl_newaxis = delta_nl_reshaped[:, :, :, np.newaxis, :, np.newaxis]
        delta_nl_broadcast, _ = np.broadcast_arrays(delta_nl_newaxis, reshaped_delta)

        # Get delta terms only from values that passed maxpooling
        reshaped_delta[mask] = delta_nl_broadcast[mask]
        reshaped_delta /= np.sum(mask, axis=(3, 5), keepdims=True)

        # Determine delta term for this layer
        trans_input_shape = self.input.transpose(3, 2, 0, 1).shape
        self.delta = reshaped_delta.reshape(trans_input_shape)
        self.delta = self.delta.transpose(2, 3, 1, 0)

    # Source: https://github.com/yunjey/cs231n/
