import numpy as np

class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, n_filters, kernel, activation, activation_prime):
        # Layer input and its shape
        self.input = None
        self.input_shape = input_shape

        # Number of filters used in the layer and their shape
        self.n_filters = n_filters
        self.kernel_shape = kernel.shape

        # Initialization of filters using normal distribution
        standard_dev = 1 / np.sqrt(np.prod(self.kernel_shape))
        self.filters = np.random.normal(0, standard_dev, self.kernel_shape)

        # Layer activation function and its derivative
        self.activation = activation
        self.activation_prime = activation_prime

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None
        self.delta_weights = 0
        self.delta_biases = 0

    def forward_prop(self, layer_input):
        raise NotImplementedError

    def backward_prop(self, next_layer):
        raise NotImplementedError