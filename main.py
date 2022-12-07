import numpy as np
from conv_layer import ConvolutionalLayer


def main():
    test = np.array([i for i in range(64)]).reshape(8, 8)
    test_cl = ConvolutionalLayer()
    kernel = np.ones((3, 3))

    test_cl.set_filter_and_activation_fun(kernel, 5)
    test_cl.compute_output(test)


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
