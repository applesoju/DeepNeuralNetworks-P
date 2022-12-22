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

POOLING_LAYER_KERNEL_SHAPE = (2, 2)
POOLING_LAYER_STRIDES_NUMBER = 2
DROPOUT_LAYER_PROBABILITY = 0.25


def layers_test():
    test = np.array([i for i in range(64)]).reshape(8, 8) / 64

    test_cl = ConvolutionalLayer(
        kernel=np.ones((3, 3))
    )
    out = test_cl.compute_output(test)
    print(f'Convolutional:\n{out}')

    test_al = ActivationLayer(
        activ_fun=af.relu
    )
    out = test_al.compute_output(out)
    print(f'Activation:\n{out}')

    test_pl = PoolingLayer(
        kernel_shape=POOLING_LAYER_KERNEL_SHAPE,
        n_strides=POOLING_LAYER_STRIDES_NUMBER
    )
    out = test_pl.compute_output(out)
    print(f'Pooling:\n{out}')

    test_fl = FlatteningLayer()
    out = test_fl.compute_output(out)
    print(f'Flattening:\n{out}')

    test_dl = DenseLayer(
        input_size=len(out),
        output_size=32
    )
    out = test_dl.compute_output(out)
    print(f'Dense:\n{out}')

    test_dol = DropoutLayer(
        probability=DROPOUT_LAYER_PROBABILITY
    )
    out = test_dol.compute_output(out)
    print(f'Dropout:\n{out}')


def network_test():
    cnn = Network()

    test_image = cv2.imread(
        'images/NonDemented/26.jpg',
        cv2.IMREAD_GRAYSCALE
    )

    convo_kernel_list = [
        np.ones((7, 7)) / 49,
        np.ones((5, 5)) / 25,
        np.ones((3, 3)) / 9
    ]

    neuron_count_in_dense = [
        int(test_image.size / (4 ** 3)),    # 3 pooling layers with 2x2 kernels
        1024,
        4                                   # output layer
    ]

    layer_list = [
        ConvolutionalLayer(
            kernel=convo_kernel_list[0]
        ),
        ActivationLayer(
            activ_fun=af.relu
        ),
        PoolingLayer(
            kernel_shape=(2, 2),
            n_strides=2
        ),
        ConvolutionalLayer(
            kernel=convo_kernel_list[1]
        ),
        ActivationLayer(
            activ_fun=af.relu
        ),
        PoolingLayer(
            kernel_shape=(2, 2),
            n_strides=2
        ),
        ConvolutionalLayer(
            kernel=convo_kernel_list[2]
        ),
        ActivationLayer(
            activ_fun=af.relu
        ),
        PoolingLayer(
            kernel_shape=(2, 2),
            n_strides=2
        ),
        FlatteningLayer(),
        DenseLayer(
            input_size=neuron_count_in_dense[0],
            output_size=neuron_count_in_dense[1],
        ),
        ActivationLayer(
            activ_fun=af.relu
        ),
        DropoutLayer(
            probability=0.25
        ),
        DenseLayer(
            input_size=neuron_count_in_dense[1],
            output_size=neuron_count_in_dense[2]
        ),
        ActivationLayer(
            af.sigmoid
        )
    ]

    for layer in layer_list:
        cnn.add(layer)

    out = cnn.train(test_image)

    print(out)


if __name__ == '__main__':
    network_test()

# Images source: https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
