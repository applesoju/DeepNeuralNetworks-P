import numpy as np

class DropoutLayer:
    def __init__(self, input_size, probability):
        # Layer input and its size
        self.input = None
        self.input_size = input_size

        # Probability of deactivating a neuron
        self.probability = probability

        # Layer output and the number of neurons in the layer (size of the output)
        self.output = None
        self.output_size = input_size

        # Prepare delta variable for backpropagation
        self.delta = None

    def forward_prop(self, layer_input, training=True):
        self.input = layer_input
        self.output = np.copy(layer_input)

        if training:
            # Choose neurons to deactivate
            amount_to_deactivate = int(self.probability * self.input_size)
            chosen_neurons = np.random.choice(self.input_size,
                                              amount_to_deactivate,
                                              replace=False)
            # Perform dropout
            self.output[chosen_neurons] = 0

            return self.output

        self.output = layer_input / self.probability
        return self.output

    def backward_prop(self, next_layer):
        self.delta = next_layer.delta
        self.delta[self.output == 0] = 0
