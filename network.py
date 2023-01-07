class Network:
    def __init__(self):
        # Layers and batch size
        self.layers = []
        self.batch_size = 0

        # Learning rate and momentum
        self.learning_rate = 0.01
        self.momentum = 0.0001

        # Loss and accuracy of training and validation sets
        self.training_loss = {}
        self.training_accuracy = {}

        self.validation_loss = {}
        self.validation_accuracy = {}

    def add(self, layer):
        raise NotImplementedError

    def train(self, inputs, correct_outputs, epochs, batch_size, shuffle=False, validation_split = 0.2):
        raise NotImplementedError

    def forward_propagation(self, inputs, training=True):
        raise NotImplementedError

    def categorical_cross_entropy_loss(self, correct_outputs, network_outputs):
        raise NotImplementedError

    def backward_propagation(self, loss, adjust_params):
        raise NotImplementedError

    def classify(self, inputs):
        raise NotImplementedError