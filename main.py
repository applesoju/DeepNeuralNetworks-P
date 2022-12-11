import cv2
import numpy as np

import activ_funs as af
from activ_layer import ActivationLayer
from conv_layer import ConvolutionalLayer
from dens_layer import DenseLayer
from drop_layer import DropoutLayer
from flat_layer import FlatteningLayer
from network import Network
from pool_layer import PoolingLayer


def layers_test():
    test = np.array([i for i in range(64)]).reshape(8, 8) / 64
    test_cl = ConvolutionalLayer()
    test_al = ActivationLayer()
    test_pl = PoolingLayer()
    test_fl = FlatteningLayer()
    test_dl = DenseLayer()
    test_dol = DropoutLayer()

    kernel = np.ones((3, 3))
    test_cl.set_filters(kernel)
    out = test_cl.compute_output(test)
    print(f'Convolutional:\n{out}')

    test_al.set_activation_function(af.relu)
    out = test_al.compute_output(out)
    print(f'Activation:\n{out}')

    test_pl.set_kernel_and_strides((2, 2), 2)
    out = test_pl.compute_output(out)
    print(f'Pooling:\n{out}')

    out = test_fl.compute_output(out)
    print(f'Flattening:\n{out}')

    test_dl.initialize_weights_and_bias(len(out), 32)
    out = test_dl.compute_output(out)
    print(f'Dense:\n{out}')

    test_dol.set_propability(0.25)
    out = test_dol.compute_output(out)
    print(f'Dropout:\n{out}')


def network_test():
    cnn = Network()
    cnn.add(ConvolutionalLayer())  # 7x7
    cnn.add(ActivationLayer())  # ReLU
    cnn.add(PoolingLayer())  # 2x2

    cnn.add(ConvolutionalLayer())  # 5x5
    cnn.add(ActivationLayer())  # ReLU
    cnn.add(PoolingLayer())  # 2x2

    cnn.add(ConvolutionalLayer())  # 3x3
    cnn.add(ActivationLayer())  # ReLU
    cnn.add(PoolingLayer())  # 2x2

    cnn.add(FlatteningLayer())  # Flatten
    cnn.add(DenseLayer())  # 1024 neurons
    cnn.add(ActivationLayer())  # ReLU
    cnn.add(DropoutLayer())  # 0.25 dropout rate

    cnn.add(DenseLayer())  # 4 neurons
    cnn.add(ActivationLayer())  # Sigmoid

    test_image = cv2.imread('images/NonDemented/26.jpg', cv2.IMREAD_GRAYSCALE)

    filter_1 = np.ones((7, 7)) / 49
    filter_2 = np.ones((5, 5)) / 25
    filter_3 = np.ones((3, 3)) / 9

    cnn.layers[0].set_filters(filter_1)
    cnn.layers[1].set_activation_function(af.relu)
    cnn.layers[2].set_kernel_and_strides((2, 2), 2)

    cnn.layers[3].set_filters(filter_2)
    cnn.layers[4].set_activation_function(af.relu)
    cnn.layers[5].set_kernel_and_strides((2, 2), 2)

    cnn.layers[6].set_filters(filter_3)
    cnn.layers[7].set_activation_function(af.relu)
    cnn.layers[8].set_kernel_and_strides((2, 2), 2)

    dl_input_size = int(test_image.size / (4 ** 3))
    cnn.layers[10].initialize_weights_and_bias(dl_input_size, 1024)
    cnn.layers[11].set_activation_function(af.relu)
    cnn.layers[12].set_propability(0.25)

    cnn.layers[13].initialize_weights_and_bias(1024, 4)
    cnn.layers[14].set_activation_function(af.sigmoid)

    out = cnn.train(test_image)
    print(out)


if __name__ == '__main__':
    network_test()

# Images source: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
