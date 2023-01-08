import cv2
import numpy as np

from layers import funs
from layers.conv_layer import ConvolutionalLayer
from layers.dens_layer import DenseLayer
from layers.drop_layer import DropoutLayer
from layers.flat_layer import FlatteningLayer
from layers.pool_layer import MaxPoolingLayer
from network import Network


def dense_forward_test(test_input=None, n=4, act=funs.relu, act_d=funs.relu_prime):
    test_input = np.random.rand(8) if test_input is None else test_input

    dense = DenseLayer(input_shape=test_input.shape,
                       n_neurons=n,
                       activation=act,
                       activation_deriv=act_d)

    test_output = dense.forward_prop(test_input)

    print(f'Input:\n{test_image}')
    print(f'Weights:\n{dense.weights}')
    print(f'Biases:\n{dense.biases}')
    print(f'Output:\n {test_output}')

    return dense


def convo_forward_test(test_input=None, n=1, plot=False):
    test_image = cv2.imread('images/test/NonDemented/26.jpg',
                            cv2.IMREAD_GRAYSCALE) if test_input is None else test_input
    convo = ConvolutionalLayer(input_shape=test_image.shape,
                               n_filters=n,
                               activation=funs.relu,
                               activation_deriv=funs.relu_prime)
    test_output = convo.forward_prop(test_image)

    if plot:
        for filt in range(test_output.shape[-1]):
            plt.imshow(test_output[:, :, filt])
            plt.show()

    print(f'Input:\n{test_image}')
    print(f'Weights:\n{convo.weights}')
    print(f'Biases:\n{convo.biases}')
    print(f'Output:\n {test_output}')

    return convo


def drop_forward_test(test_input=None):
    test_input = np.random.random((1, 20)) if test_input is None else test_input
    drop = DropoutLayer(test_input.shape,
                        0.2)
    test_output = drop.forward_prop(test_input)
    difference = test_input - test_output

    print(f'Input:\n{test_input}')
    print(f'Output:\n {test_output}')
    print(f'Difference:\n {difference}')

    return drop


def pool_forward_test(test_input=None, plot=False):
    test_image = cv2.imread('images/test/NonDemented/26.jpg',
                            cv2.IMREAD_GRAYSCALE) if test_input is None else test_input
    test_image = np.atleast_3d(test_image)

    pool = MaxPoolingLayer(input_shape=test_image.shape)
    test_output = pool.forward_prop(test_image)

    if plot:
        for filt in range(test_output.shape[-1]):
            plt.imshow(test_output[:, :, filt])
            plt.show()

    print(f'Input:\n{test_image}')
    print(f'Output:\n {test_output}')

    return pool


def flat_forward_test(test_input=None):
    test_input = np.random.random((5, 5)) if test_input is None else test_input
    test_input = np.atleast_3d(test_input)

    flat = FlatteningLayer(input_shape=test_input.shape)
    test_output = flat.forward_prop(test_input)

    print(f'Input:\n{test_input}')
    print(f'Output:\n{test_output}')

    return flat


if __name__ == '__main__':
    # img = cv2.imread('images/test/NonDemented/26.jpg', cv2.IMREAD_GRAYSCALE)
    #
    # normalized = (img - img.mean())/img.std()
    #
    # print(f'\nForward propagation begins.\n')
    #
    # c = convo_forward_test(normalized, 4)
    # cc = convo_forward_test(c.output, 4)
    # p = pool_forward_test(cc.output)
    # f = flat_forward_test(p.output)
    # de = dense_forward_test(f.output, 144)
    # dr = drop_forward_test(de.output)
    # dede = dense_forward_test(dr.output, 12)
    # out = dense_forward_test(dede.output, 4, funs.softmax, funs.softmax_prime)
    #
    # correct_values = np.array([1.0, 0.0, 0.0, 0.0]).reshape(1, -1)
    # mse = funs.mse(correct_values, out.output)
    #
    # print(f'MSE:\n{mse}')
    #
    # layers = [c, cc, p, f, de, dr, dede, out]
    #
    # loss = correct_values - out.output
    # loss /= layers[-1].activation_deriv(out.output)
    #
    # print(f'\nBackward propagation begins.\n')
    #
    # for i in reversed(range(len(layers))):
    #     layer = layers[i]
    #
    #     if layer == layers[-1]:
    #         layer.error = loss
    #         layer.delta = layer.error * layer.activation_deriv(layer.output)
    #
    #         layer.delta_weights += layer.delta * np.atleast_2d(layer.input).T
    #         layer.delta_biases += layer.delta
    #
    #     else:
    #         downstream_layer = layers[i + 1]
    #         layer.backward_prop(downstream_layer)
    #
    #         if i in (0, 1):
    #             for delta_ in range(layer.delta.shape[-1]):
    #                 plt.imshow(layer.delta[:, :, delta_])
    #                 plt.show()
    #
    #
    #     print(f'Layer {i} delta:\n{layer.delta}')

    img = cv2.imread('images/test/NonDemented/26.jpg', cv2.IMREAD_GRAYSCALE)

    correct_values = np.array([1.0, 0.0, 0.0, 0.0])

    layer_list = [
        # Convolutional, 7x7, 16 filters, ReLU
        ConvolutionalLayer(input_shape=(208, 176),
                           n_filters=16,
                           kernel_shape=(7, 7),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling, 2x2
        MaxPoolingLayer(input_shape=(208, 176, 16)),

        # Convolutional, 5x5, 16 filters, ReLU
        ConvolutionalLayer(input_shape=(104, 88, 16),
                           n_filters=16,
                           kernel_shape=(5, 5),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling 2x2
        MaxPoolingLayer(input_shape=(104, 88, 16)),

        # Convolutional, 3x3, 8 filters, ReLU
        ConvolutionalLayer(input_shape=(52, 44, 16),
                           n_filters=8,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling 2x2
        MaxPoolingLayer(input_shape=(52, 44, 8)),

        # Flattening to (1, n)
        FlatteningLayer(input_shape=(26, 22, 8)),

        # Dense, 1024, ReLU
        DenseLayer(input_shape=(1, 22 * 26 * 8),
                   n_neurons=1024,
                   activation=funs.relu,
                   activation_deriv=funs.relu_prime),

        # Dropout 25%
        DropoutLayer(input_shape=(1, 1024),
                     probability=0.25),

        # Dense, 4, SoftMax
        DenseLayer(input_shape=(1, 1024),
                   n_neurons=4,
                   activation=funs.softmax,
                   activation_deriv=funs.softmax_prime)
    ]

    simple_ll = [
        # Convolutional, 3x3, 16 filters, ReLU
        ConvolutionalLayer(input_shape=(208, 176),
                           n_filters=4,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling, 2x2
        MaxPoolingLayer(input_shape=(208, 176, 4)),

        # Flattening to (1, n)
        FlatteningLayer(input_shape=(104, 88, 4)),

        # Dense, 1024, ReLU
        DenseLayer(input_shape=(1, 104 * 88 * 4),
                   n_neurons=1024,
                   activation=funs.relu,
                   activation_deriv=funs.relu_prime),

        # Dropout 25%
        DropoutLayer(input_shape=(1, 1024),
                     probability=0.25),

        # Dense, 4, SoftMax
        DenseLayer(input_shape=(1, 1024),
                   n_neurons=4,
                   activation=funs.softmax,
                   activation_deriv=funs.softmax_prime)
    ]

    cnn = Network()

    for lay in simple_ll:
        cnn.add(lay)

    cnn.compile()

    prediction = cnn.forward_propagation(img)
    output_values = np.squeeze(prediction)

    l, e = cnn.cross_entropy_loss(correct_values, output_values)

    cnn.backward_propagation(l, True)

    print(prediction)
