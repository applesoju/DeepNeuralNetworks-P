import time

import numpy as np
import pandas as pd

from adam import AdamOptimizer
from layers import funs


class Network:
    def __init__(self):
        # Layers list
        self.layers = []

        # Number of epochs and batch size
        self.epochs = 1
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
        next_input_shape = None

        for layer in self.layers:
            if layer.input_shape is None:
                layer.init_params(next_input_shape)

            next_input_shape = layer.output_shape
            print(next_input_shape)
        self.optimizer = AdamOptimizer(self.layers)
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

        # Categorical Cross Entropy
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
            if type(layer).__name__ in ['DenseLayer', 'ConvolutionalLayer']:
                layer.delta_weights = np.zeros(layer.delta_weights.shape)
                layer.delta_biases = np.zeros(layer.delta_biases.shape)

            if type(layer).__name__ == 'ConvolutionalLayer':
                layer.col_delta_weights = np.zeros(layer.col_delta_weights.shape)

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
        # Model is not compiled
        if not self.is_compiled:
            raise ValueError('Model not compiled.')

        # The number of inputs and labels doesn't match
        if len(inputs) != len(labels):
            raise ValueError('Lenght of labels and training input is not the same.')

        # The input shape and first layer input shape don't match
        if inputs[0].shape != self.layers[0].input_shape[0: 2]:
            insh = inputs[0].shape
            lash = self.layers[0].input_shape[0: 2]

            raise ValueError(f'An input of shape {insh} was given,'
                             f'while the network expects input in shape {lash}.')

        # The size of labels and the size of network output don't match
        if labels.shape[-1] != self.layers[-1].n_neurons:
            lash = labels.shape[-1]
            nesh = self.layers[-1].n_neurons

            raise ValueError(f'A labels vector of shape {lash} was given,'
                             f'while the network outputs vector in shape {nesh}')

    def train(self, inputs, correct_outputs, epochs, batch_size, shuffle=False, validation_split=0.2):
        self.check_for_training(inputs, correct_outputs)

        self.epochs = epochs
        self.batch_size = batch_size

        start_time = time.time()

        indices = np.arange(0, len(inputs), dtype=np.int32)
        if shuffle:
            np.random.shuffle(indices)

        validation_data_len = int(len(inputs) * validation_split)
        validation_indices = []

        while len(validation_indices) != validation_data_len:
            random_index = np.random.randint(0, len(inputs))

            if random_index not in validation_indices:
                validation_indices.append(random_index)

        validation_indices = np.array(validation_indices)
        validation_inputs = inputs[validation_indices]
        validation_labels = correct_outputs[validation_indices]

        indices = np.array([i for i in indices if i not in validation_indices])

        n_batches = int(len(indices) / batch_size)
        if len(indices) % batch_size != 0:
            n_batches += 1

        print(f'Total number of images: {len(inputs)}\n'
              f'Number of training samples: {len(indices)}\n'
              f'Number of validation saples: {len(validation_indices)}\n'
              f'Number of batches: {n_batches}\n'
              f'Size of one batch: {batch_size}')

        batches = np.array_split(indices, n_batches)

        for epoch in range(epochs):
            errors = []

            for batch_num, batch in enumerate(batches):
                batch_loss = 0
                xs, ys = inputs[batch], correct_outputs[batch]

                # make an exception for ConvolutionalLayers, so they can process whole batch in one go
                batch_t = time.time()
                for i, xy in enumerate(zip(xs, ys)):
                    x, y = xy

                    output = self.forward_propagation(x)
                    out = np.squeeze(output)
                    loss, error = self.cross_entropy_loss(correct_output=y, network_output=out)

                    batch_loss += loss
                    errors.append(error)

                    update = False
                    if i == batch_size - 1:
                        update = True
                        loss = batch_loss / batch_size

                    self.backward_propagation(loss, update)

                print(f'Batch {batch_num} of Epoch {epoch} done in {round(time.time() - batch_t, 2)}.')
                batch_t = time.time()

            train_output = self.classify(inputs[indices])
            train_loss, train_error = self.cross_entropy_loss(correct_outputs[indices], train_output)

            validation_output = self.classify(validation_inputs)
            validation_loss, validation_error = self.cross_entropy_loss(validation_labels, validation_output)

            train_accuracy = np.squeeze(train_output.argmax(axis=2)) == correct_outputs[indices].argmax(axis=1)
            validation_accuracy = np.squeeze(validation_output.argmax(axis=2)) == validation_labels.argmax(axis=1)

            self.training_loss[epoch] = round(train_error.mean(), 4)
            self.training_accuracy[epoch] = round(train_accuracy.mean() * 100, 4)
            self.validation_loss[epoch] = round(validation_error.mean(), 4)
            self.validation_accuracy[epoch] = round(validation_accuracy.mean() * 100, 4)

            print(f'Epoch {epoch}:\n'
                  f'Time: {round(time.time() - start_time, 3)} seconds\n'
                  f'Train loss: {round(train_error.mean(), 4)}\n'
                  f'Train accuracy: {round(train_accuracy.mean() * 100, 4)}%\n'
                  f'Validation loss: {(round(validation_error.mean(), 4))}\n'
                  f'Validation Accuracy: {round(validation_accuracy.mean() * 100, 4)}%\n')

            start_time = time.time()

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

        return np.array(predictions)

    def summary(self):
        # Model is not compiled
        if not self.is_compiled:
            raise ValueError('Model needs to be compiled.')

        layer_name = []
        layer_input = []
        layer_output = []
        layer_activation = []

        model_df = None

        for layer in self.layers:
            layer_name.append(layer.name)
            layer_input.append(layer.input_shape)
            layer_output.append(layer.output_shape)
            try:
                layer_activation.append(layer.activation.__name__)
            except AttributeError:
                layer_activation.append('None')

            model_dict = {
                'Layer Name': layer_name,
                'Input': layer_input,
                'Output': layer_output,
                'Activation Function': layer_activation
            }
            model_df = pd.DataFrame(model_dict).set_index("Layer Name")

        print(model_df)

    def save_to_json(self, path='model.json'):
        dict_model = {'model': str(type(self).__name__)}

        to_save = ['name', 'n_neurons', 'input_shape', 'output_shape',
                   'wights', 'biases', 'activation', 'n_filters',
                   'kernel_shape', 'padding', 'probability']


