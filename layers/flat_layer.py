from .layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        super().__init__()

        self.input_shape = None

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_flattening()
        self.input_shape = input_data.shape
        return self.outputs

    def perform_flattening(self):
        output = self.inputs.flatten()
        return output

    def perform_backward_prop(self, output_err, learn_rate):
        return self.outputs.reshape(self.input_shape)