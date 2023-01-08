import numpy as np


class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.01,
                 beta1=0.9, beta2=0.999, training=True):
        # Layers of network
        self.layers = layers

        # Learning rate
        self.learning_rate = learning_rate

        # Beta1 and Beta2
        self.beta1 = beta1
        self.beta2 = beta2

        # Boolean value denoting the process of training
        self.training = training

        # Small value for prevention of division by 0
        self.eps = 1e-15

        self.ts = [0] * len(layers)

        self.weights_adam1 = [0] * len(layers)
        self.weights_adam2 = [0] * len(layers)

        self.biases_adam1 = [0] * len(layers)
        self.biases_adam2 = [0] * len(layers)

    def reset_adam_params(self):
        for i in range(len(self.ts)):
            self.ts[i] = 0
            self.weights_adam1[i] = 0
            self.weights_adam2[i] = 0
            self.biases_adam1[i] = 0
            self.biases_adam2[i] = 0

    def compute_moment(self, beta, layer_index):
        i = layer_index
        # Beta1
        if beta == self.beta1:
            weights_moments = beta * self.weights_adam1[i] + (1 - beta) * self.layers[i].delta_weights
            biases_moments = beta * self.biases_adam1[i] + (1 - beta) * self.layers[i].biases
        # Beta 2
        else:
            weights_moments = beta * self.weights_adam2[i] + (1 - beta) * (self.layers[i].delta_weights ** 2)
            biases_moments = beta * self.biases_adam2[i] + (1 - beta) * (self.layers[i].biases ** 2)

        return weights_moments, biases_moments

    def get_mcaps(self, layer_index, t):
        weights_mcap = self.weights_adam1[layer_index] / (1 - self.beta1 ** t)
        biases_mcap = self.biases_adam1[layer_index] / (1 - self.beta1 ** t)

        return weights_mcap, biases_mcap

    def get_vcaps(self, layer_index, t):
        weights_vcap = self.weights_adam2[layer_index] / (1 - self.beta2 ** t)
        biases_vcap = self.biases_adam2[layer_index] / (1 - self.beta2 ** t)

        return weights_vcap, biases_vcap

    def adam(self):
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, 'weights'):
                continue

            if not self.training:
                self.reset_adam_params()

            else:
                self.ts[i] += 1
                t = self.ts[i]

                # Get weights' and biases' moments
                self.weights_adam1[i], self.biases_adam1[i] = self.compute_moment(self.beta1, i)
                self.weights_adam2[i], self.biases_adam2[i] = self.compute_moment(self.beta2, i)

                # Get needed parameters
                w_mcap, b_mcap = self.get_mcaps(i, t)
                w_vcap, b_vcap = self.get_vcaps(i, t)

                # Adjust weights
                layer.delta_weights = w_mcap / (np.sqrt(w_vcap) + self.eps)
                layer.weights += self.learning_rate * layer.delta_weights

                # Adjust biases
                layer.delta_biases = b_mcap / (np.sqrt(b_vcap) + self.eps)
                layer.biases += self.learning_rate * layer.delta_biases
