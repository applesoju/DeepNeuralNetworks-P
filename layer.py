class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def compute_output(self, inputs):
        self.inputs = inputs
