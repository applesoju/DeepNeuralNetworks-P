class Network:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def train(self):
        raise NotImplementedError

    def classify(self, input_layer):
        raise NotImplementedError
