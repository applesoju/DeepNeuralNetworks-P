import cv2
import numpy as np

from layers import funs
from layers.conv_layer import ConvolutionalLayer
from layers.dens_layer import DenseLayer
# from layers.drop_layer import DropoutLayer
# from layers.flat_layer import FlatteningLayer
# from layers.pool_layer import PoolingLayer
from network import Network

# POOLING_LAYER_KERNEL_SHAPE = (2, 2)
# POOLING_LAYER_STRIDES_NUMBER = 2
# DROPOUT_LAYER_PROBABILITY = 0.25
#
#
# def layers_test():
#     test = np.array([i for i in range(64)]).reshape(8, 8) / 64
#
#     test_cl = ConvolutionalLayer(
#         kernel=np.ones((3, 3))
#     )
#     out = test_cl.compute_output(test)
#     print(f'Convolutional:\n{out}')
#
#     test_al = ActivationLayer(
#         activ_fun=funs.relu,
#         activ_fun_deriv=funs.relu_prime
#     )
#     out = test_al.compute_output(out)
#     print(f'Activation:\n{out}')
#
#     test_pl = PoolingLayer(
#         kernel_shape=POOLING_LAYER_KERNEL_SHAPE,
#         n_strides=POOLING_LAYER_STRIDES_NUMBER
#     )
#     out = test_pl.compute_output(out)
#     print(f'Pooling:\n{out}')
#
#     test_fl = FlatteningLayer()
#     out = test_fl.compute_output(out)
#     print(f'Flattening:\n{out}')
#
#     test_dl = DenseLayer(,
#     out = test_dl.compute_output(out)
#     print(f'Dense:\n{out}')
#
#     test_dol = DropoutLayer(
#         probability=DROPOUT_LAYER_PROBABILITY
#     )
#     out = test_dol.compute_output(out)
#     print(f'Dropout:\n{out}')
#
#
# def network_test():
#     cnn = Network(funs.mse, funs.mse_prime)
#
#     test_image = cv2.imread(
#         'images/test/NonDemented/26.jpg',
#         cv2.IMREAD_GRAYSCALE
#     )
#
#     correct_values = np.array([1.0,  # NonDemented
#                                0.0,  # VeryMildDemented
#                                0.0,  # MildDemented
#                                0.0])  # ModerateDemented
#
#     convo_kernel_list = [
#         np.ones((7, 7)) / 49,
#         np.ones((5, 5)) / 25,
#         np.ones((3, 3)) / 9
#     ]
#
#     neuron_count_in_dense = [
#         int(test_image.size / (4 ** 3)),  # 3 pooling layers with 2x2 kernels
#         1024,
#         4  # output layer
#     ]
#
#     layer_list = [
#         ConvolutionalLayer(
#             kernel=convo_kernel_list[0]
#         ),
#         ActivationLayer(
#             activ_fun=funs.relu,
#             activ_fun_deriv=funs.relu_prime
#         ),
#         PoolingLayer(
#             kernel_shape=(2, 2),
#             n_strides=2
#         ),
#         ConvolutionalLayer(
#             kernel=convo_kernel_list[1]
#         ),
#         ActivationLayer(
#             activ_fun=funs.relu,
#             activ_fun_deriv=funs.relu_prime
#         ),
#         PoolingLayer(
#             kernel_shape=(2, 2),
#             n_strides=2
#         ),
#         ConvolutionalLayer(
#             kernel=convo_kernel_list[2]
#         ),
#         ActivationLayer(
#             activ_fun=funs.relu,
#             activ_fun_deriv=funs.relu_prime
#         ),
#         PoolingLayer(
#             kernel_shape=(2, 2),
#             n_strides=2
#         ),
#         FlatteningLayer(),
#         DenseLayer(,,
#         ActivationLayer(
#             activ_fun=funs.relu,
#             activ_fun_deriv=funs.relu_prime
#         ),
#         DropoutLayer(
#             probability=0.25
#         ),
#         DenseLayer(,,
#         ActivationLayer(
#             activ_fun=funs.sigmoid,
#             activ_fun_deriv=funs.sigmoid_prime
#         )
#     ]
#
#     for layer in layer_list:
#         cnn.add(layer)
#
#     out = cnn.train(input_layer=test_image,
#                     correct_output=correct_values)
#
#     print(out)
def dense_test():
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

    return

def convo_test():
    test_input = np.random.random((4, 4,)) * 12
    convo = ConvolutionalLayer(input_shape=(4, 4),
                               n_filters=2,
                               activation=funs.relu)
    test_output = convo.forward_prop(test_input)

    print(f'Input:\n{test_input}')
    print(f'Weights:\n{convo.weights}')
    print(f'Biases:\n{convo.biases}')
    print(f'Output:\n {test_output}')

    return

if __name__ == '__main__':
    # dense_test()
    convo_test()