class Network:
    def __init__(self, loss_fun, deriv_loss_fun):
        self.layers = []
        self.loss = loss_fun
        self.loss_prime = deriv_loss_fun

    def add(self, layer):
        self.layers.append(layer)

    def train(self, input_layer, correct_output, epochs=None, learn_rate=None):
        input_layer = input_layer.astype('float64')
        output = input_layer / 255.0

        for layer in self.layers:
            print(f'Layer shape: {output.shape}')
            output = layer.compute_output(output)

        print(f'Output layer: {output}')

        output_error = self.loss_prime(correct_output, output)

        print(f'Output Error: {output_error}')

        for layer in self.layers[::-1]:
            print(output_error.shape)

            output_error = layer.perform_backward_prop(output_error, 0.05)

        return output

    def classify(self, input_layer):
        raise NotImplementedError
