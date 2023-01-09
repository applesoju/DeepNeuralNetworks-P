import cv2
import numpy as np
import os
from layers import funs
from layers.conv_layer import ConvolutionalLayer
from layers.dens_layer import DenseLayer
from layers.drop_layer import DropoutLayer
from layers.flat_layer import FlatteningLayer
from layers.pool_layer import MaxPoolingLayer
from network import Network
import time

def prepare_data(dir_path_to_data, n_samples=0):
    if not os.path.exists(dir_path_to_data):
        raise FileNotFoundError(f'Directory {dir_path_to_data} does not exist.')

    # Get directories which correspond to classes
    classes = next(os.walk(dir_path_to_data))[1]

    label_list = []
    image_list = []

    for class_index, class_name in enumerate(classes):
        start_time = time.time()
        class_dir_path = f'{dir_path_to_data}/{class_name}'

        samples = os.listdir(class_dir_path)[:n_samples] if n_samples > 0 else os.listdir(class_dir_path)

        for file in samples:
            img_path = f'{class_dir_path}/{file}'
            img = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

            label_list.append(class_index)
            image_list.append(img)

        print(f'Loading images from class {class_name} done in {round(time.time() - start_time, 3)} seconds.')

    label_array = np.array(label_list)
    image_array = np.array(image_list)

    return image_array, label_array


if __name__ == '__main__':
    final_model = [
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

    x, y = prepare_data('images/test', 64)
    layers_list = final_model

    cnn = Network()

    for lay in layers_list:
        cnn.add(lay)

    cnn.compile()

    cnn.train(inputs=x,
              correct_outputs=y,
              epochs=10,
              batch_size=16,
              shuffle=True,
              validation_split=0.25)

    prediction = cnn.forward_propagation(img)
    output_values = np.squeeze(prediction)

    l, e = cnn.cross_entropy_loss(correct_values, output_values)

    cnn.backward_propagation(l, True)

    print(prediction)
