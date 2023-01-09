from layers import funs
import numpy as np

from adam import AdamOptimizer


class Network:
    def __init__(self):
        # Layers and batch size
        self.layers = []
        self.batch_size = 1

        # Learning rate and optimizer
        self.learning_rate = 0.01
        self.optimizer = None

        # Loss and accuracy of training and validation sets
        self.training_loss = {}
        self.training_accuracy = {}

        self.validation_loss = {}
        self.validation_accuracy = {}

        self.is_compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        self.optimizer = AdamOptimizer(self.layers)

        # next_input_shape = None

        # for layer in self.layers:         # TODO: automatic input/output size (shape) determination
        #     if layer.input_shape is None:
        #         layer.input_shape = next_input_shape
        #         layer.get_output_shape()
        #
        #     next_input_shape = layer.output
        self.is_compiled = True

    def forward_propagation(self, inputs, training=True):
        next_input = inputs
        out = None

        for layer in self.layers:
            # If a layer is a DropoutLayer then pass training argument
            #   so the neurons will be dropped only during training
            if type(layer).__name__ == 'DropoutLayer':
                out = layer.forward_prop(next_input, training)

            else:
                out = layer.forward_prop(next_input)

            next_input = out

        return out

    def cross_entropy_loss(self, correct_output, network_output):
        np.errstate(divide='ignore')

        # Binary Cross Entropy
        if len(correct_output) == len(network_output) == 1:
            err = -(correct_output * np.log(network_output) +
                    (1 - correct_output) * np.log(1 - network_output))
            loss = -(correct_output / network_output -
                     (1 - correct_output) / (1 - network_output))

        # Categorical Cross Entropy
        else:
            err = -np.sum((correct_output * np.nan_to_num(np.log(network_output)) +
                           (1 - correct_output) * np.nan_to_num(np.log(1 - network_output))))

            # SoftMax output
            if self.layers[-1].activation == funs.softmax:
                loss = correct_output - network_output
                loss /= self.layers[-1].activation_deriv(network_output)

            # Other output
            else:
                correct_output = np.float64(correct_output)
                network_output += 1e-15

                loss = -(np.nan_to_num(correct_output / network_output) -
                         np.nan_to_num((1 - correct_output) / (1 - network_output)))

        return loss, err

    def reset_gradients(self):
        for layer in self.layers:
            # Reset gradients of weights and biases in layers that have them
            try:
                layer.delta_weights = np.zeros(layer.delta_weights.shape)
                layer.delta_biases = np.zeros(layer.delta_biases.shape)

            except AttributeError:
                pass

    def backward_propagation(self, loss, adjust_params):  # TODO: add comments and test
        for i in reversed(range(len(self.layers))):
            # Current layer
            layer = self.layers[i]

            # If it's the output layer set its error and delta terms
            if layer == self.layers[-1]:
                layer.error = loss
                layer.delta = layer.error * layer.activation_deriv(layer.output)

                layer.delta_weights += layer.delta * layer.input.T
                layer.delta_biases += layer.delta

            # If it's not then backpropagate the error
            else:
                next_layer = self.layers[i + 1]
                layer.backward_prop(next_layer)

            # If parameters should be updated then average out delta weights and biases
            if adjust_params:
                try:
                    layer.delta_weights /= self.batch_size
                    layer.delta_biases /= self.batch_size

                except AttributeError:
                    pass

        # If parameters should be updated then use the optimizer and reset gradient values
        if adjust_params:
            self.optimizer.adam()
            self.reset_gradients()

    def check_for_training(self, inputs, labels):
        if not self.is_compiled:
            raise ValueError('Model not compiled.')

        if len(inputs) != len(labels):
            raise ValueError('Lenght of labels and training input is not the same.')

        if inputs[0].shape != self.layers[0].input_shape[0: 2]:
            insh = inputs[0].shape
            lash = self.layers[0].input_shape[0: 2]

            raise ValueError(f'An input of shape {insh} was given,'
                             f'while the network expects input in shape {lash}.')

        if labels.shape[-1] != self.layers[-1].n_neurons:
            lash = labels.shape[-1]
            nesh = self.layers[-1].n_neurons

            raise ValueError(f'A labels vector of shape {lash} was given,'
                             f'while the network outputs vector in shape {nesh}')

    def train(self, inputs, correct_outputs, epochs, batch_size, shuffle=False, validation_split=0.2):
        raise NotImplementedError

    def classify(self, input_for_classification):
        # Result saving
        predictions = []

        # Input is just one object
        if input_for_classification.shape == self.layers[0].input_shape[0: 2]:
            predictions.append(
                self.forward_propagation(input_for_classification,
                                         training=False)
            )

        # Input is a list of objects
        else:
            for one_input in input_for_classification:
                predictions.append(
                    self.forward_propagation(one_input,
                                             training=False)
                )

    def summary(self):
        raise NotImplementedError