from layer import Layer


class ConvolutionalLayer(Layer):
    def __init__(self):
        super().__init__()
        self.activation_function = None
        self.filter = None
