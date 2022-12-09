import numpy as np

from conv_layer import ConvolutionalLayer
from activ_layer import ActivationLayer
from pool_layer import PoolingLayer
import activ_funs as af


def main():
    test = np.array([i for i in range(64)]).reshape(8, 8)
    test_cl = ConvolutionalLayer()
    test_al = ActivationLayer()
    test_pl = PoolingLayer()

    kernel = np.ones((3, 3))
    test_cl.set_filters(kernel)
    out = test_cl.compute_output(test)
    print(out)

    test_al.set_activation_function(af.relu)
    out = test_al.compute_output(out)
    print(out)

    test_pl.set_kernel_and_strides((2, 2), 2)
    out = test_pl.compute_output(out)
    print(out)


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
#     Dense + ReLU with Dropout
#     Dense + sigmoid
