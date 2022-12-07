class ActivationLayer(Layer):
    def __init__(self):
        super().__init__()
        self.activation_function = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

    def perform_activation(self):
        raise NotImplementedError
