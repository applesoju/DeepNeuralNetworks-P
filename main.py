import cv2
import numpy as np
import matplotlib.pyplot as plt

from layers import funs
from layers.conv_layer import ConvolutionalLayer
from layers.dens_layer import DenseLayer
from layers.drop_layer import DropoutLayer
from layers.flat_layer import FlatteningLayer
from layers.pool_layer import MaxPoolingLayer
from network import Network

def dense_forward_test():
    test_input = np.random.rand(4)
    dense = DenseLayer(input_size=len(test_input),
                       n_neurons=6,
                       activation=funs.relu,
                       activation_deriv=funs.relu_prime)
    test_output = dense.forward_prop(test_input)

    print(f'Input:\n{test_input}')
    print(f'Weights:\n{dense.weights}')
    print(f'Biases:\n{dense.biases}')
    print(f'Output:\n {test_output}')

    return test_output

def convo_forward_test(test_input=None):
    test_image = cv2.imread('images/test/NonDemented/26.jpg',
                            cv2.IMREAD_GRAYSCALE) if test_input is None else test_input
    convo = ConvolutionalLayer(input_shape=test_image.shape,
                               n_filters=2,
                               activation=funs.relu)
    test_output = convo.forward_prop(test_image)

    for f in range(test_output.shape[-1]):
        plt.imshow(test_output[:, :, f])
        plt.show()

    print(f'Input:\n{test_image}')
    print(f'Weights:\n{convo.weights}')
    print(f'Biases:\n{convo.biases}')
    print(f'Output:\n {test_output}')

    return test_output

def drop_forward_test(test_input=None):
    test_input = np.random.rand(20) if test_input is None else test_input
    drop = DropoutLayer(len(test_input),
                        0.5)
    test_output = drop.forward_prop(test_input)
    difference = test_input - test_output

    print(f'Input:\n{test_input}')
    print(f'Output:\n {test_output}')
    print(f'Difference:\n {difference}')

    return test_output

def pool_forward_test(test_input=None):
    test_image = cv2.imread('images/test/NonDemented/26.jpg',
                            cv2.IMREAD_GRAYSCALE) if test_input is None else test_input
    test_image = np.atleast_3d(test_image)

    pool = MaxPoolingLayer(input_shape=test_image.shape)
    test_output = pool.forward_prop(test_image)

    for f in range(test_output.shape[-1]):
        plt.imshow(test_output[:, :, f])
        plt.show()

    print(f'Input:\n{test_image}')
    print(f'Output:\n {test_output}')

    return test_output

def flat_forward_test(test_input=None):
    test_input = np.random.random((5, 5)) if test_input is None else test_input
    test_input = np.atleast_3d(test_input)

    flat = FlatteningLayer(input_shape=test_input.shape)
    test_output = flat.forward_prop(test_input)

    print(f'Input:\n{test_input}')
    print(f'Output:\n {test_output}')

    return test_output

if __name__ == '__main__':
    # dense_forward_test()
    # out = convo_forward_test()
    # drop_forward_test()
    # pool_forward_test(out)
    flat_forward_test()