import numpy as np

from conv_layer import ConvolutionalLayer
from activ_layer import ActivationLayer
from pool_layer import PoolingLayer
from flat_layer import FlatteningLayer
from dens_layer import DenseLayer
from drop_layer import DropoutLayer
import activ_funs as af


def main():
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


if __name__ == '__main__':
    main()

# Model :
#     7 * 7 Convolutional Layer some number of filters + ReLU
#     2 * 2 Pooling
#     5 * 5 Convolutional Layer some number of filters + ReLU
#     2 * 2 Pooling
#     3 * 3 Convolutional Layer some number of filters + ReLU
#     2 * 2 Pooling
#     Flattening
#     Dense (1024) + ReLU with Dropout
#     Dense + sigmoid
