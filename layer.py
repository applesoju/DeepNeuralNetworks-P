class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def compute_output(self, input_data):
        self.inputs = input_data
