class Network:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def train(self, input_layer):
        input_layer = input_layer.astype('float64')
        next_layer = input_layer / 255.0

        for layer in self.layers:
            print(next_layer.shape)
            next_layer = layer.compute_output(next_layer)

        return next_layer

    def classify(self, input_layer):
        raise NotImplementedError
