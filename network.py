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
            print(output.shape)
            output = layer.compute_output(output)

        print(output)

        output_error = self.loss_prime(correct_output, output)

        # for layer in self.layers[::-1]:
        #     print(output.shape)
        #     output = layer.perform_backward_prop()


        return output

    def classify(self, input_layer):
        raise NotImplementedError
