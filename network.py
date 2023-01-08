from layers import funs

class Network:
    def __init__(self):
        # Layers and batch size
        self.layers = []
        self.batch_size = 1

        # Learning rate and momentum
        self.learning_rate = 0.01
        self.momentum = 0.0001

        # Loss and accuracy of training and validation sets
        self.training_loss = {}
        self.training_accuracy = {}

        self.validation_loss = {}
        self.validation_accuracy = {}

    def add(self, layer):       # TODO: implement automatic input/output size (shape) determination
        self.layers.append(layer)

    def train(self, inputs, correct_outputs, epochs, batch_size, shuffle=False, validation_split = 0.2):
        raise NotImplementedError

    def forward_propagation(self, inputs, training=True):
        next_input = inputs
        out = None

        for layer in self.layers:
            if type(layer).__name__ == 'DropoutLayer':
                out = layer.forward_prop(next_input, training)

            else:
                out = layer.forward_prop(next_input)

            next_input = out

        return out

    def cross_entropy_loss(self, correct_output, network_output):
        # Binary Cross Entropy
        if len(correct_output) == len(network_output) == 1:
            err = -(correct_output * np.log(network_output) +
                    (1 - correct_output) * np.log(1 - network_output))
            loss = -(correct_output / network_output -
                     (1 - correct_output) / (1 - network_output))

        # Categorical Cross Entropy
        else:
            err = -np.sum((correct_output * np.log(network_output)) +
                          (1 - correct_output) * np.log(1 - network_output))

            # SoftMax output
            if self.layers[-1].activation == funs.softmax:
                loss = correct_output - network_output
                loss /= self.layers[-1].activation_deriv(network_output)

            else:
                correct_output = np.float64(correct_output)
                network_output += 1e-16

                loss = -(np.nan_to_num(correct_output / network_output) -
                         np.nan_to_num((1 - correct_output) / (1 - network_output)))

        return loss, err


    def backward_propagation(self, loss, adjust_params):
        raise NotImplementedError

    def classify(self, inputs):
        raise NotImplementedError