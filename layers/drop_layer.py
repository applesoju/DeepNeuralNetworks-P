import numpy as np

class DropoutLayer:
    def __init__(self, input_shape, probability):
        # Layer input and its size
        self.input = None
        self.input_shape = input_shape

        # Probability of deactivating a neuron and number of neurons
        self.probability = probability
        self.n_neurons = input_shape[1]

        # Layer output and the number of neurons in the layer (size of the output)
        self.output = None
        self.output_shape = input_shape

        # Prepare error and delta variables for backpropagation
        self.error = None
        self.delta = None

    def forward_prop(self, layer_input, training=True):
        self.input = layer_input
        self.output = np.copy(self.input)

        if training:
            # Choose neurons to deactivate
            amount_to_deactivate = int(self.probability * self.n_neurons)
            chosen_neurons = np.random.choice(self.n_neurons,
                                              amount_to_deactivate,
                                              replace=False)
            # Perform dropout
            self.output[:, chosen_neurons] = 0

            return self.output

        self.output = layer_input / self.probability
        return self.output

    def backward_prop(self, next_layer):
        self.error = np.dot(next_layer.weights, next_layer.delta.T).T
        self.delta = self.error * self.output
        self.delta[self.output == 0] = 0
