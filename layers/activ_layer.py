from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activ_fun, activ_fun_deriv):
        super().__init__()

        self.activation_function = activ_fun
        self.activation_function_deriv = activ_fun_deriv

    def compute_output(self, input_data):
        super().compute_output(input_data)

        self.outputs = self.perform_activation()
        return self.outputs

    def perform_activation(self):
        output = self.activation_function(self.inputs)

        return output

    def perform_backward_prop(self, output_err, learn_rate):
        return self.activation_function_deriv(self.inputs) * output_err
