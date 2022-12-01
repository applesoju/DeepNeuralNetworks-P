from layer import Layer


class PoolingLayer(Layer):
    def __init__(self):
        super().__init__()
        self.kernel = None
