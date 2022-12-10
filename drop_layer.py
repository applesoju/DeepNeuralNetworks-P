from layer import Layer


class DropoutLayer(Layer):
    def __init__(self):
        super().__init__()
        self.propability = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_dropout()
        return self.outputs

    def perform_dropout(self):
        raise NotImplementedError