import os
import time

import cv2
import numpy as np

from layers import funs
from layers.conv_layer import ConvolutionalLayer
from layers.dens_layer import DenseLayer
from layers.drop_layer import DropoutLayer
from layers.flat_layer import FlatteningLayer
from layers.pool_layer import MaxPoolingLayer
from network import Network


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

    image_array = np.array(image_list)

    if n_samples == 0:
        label_array = np.zeros((len(label_list), len(classes)))
    else:
        label_array = np.zeros((n_samples * len(classes), len(classes)))

    for i, j in enumerate(label_list):
        label_array[i, j] = 1

    return image_array, label_array


if __name__ == '__main__':
    final_model = [
        # Convolutional, 7x7, 16 filters, ReLU
        ConvolutionalLayer(input_shape=(200, 200),
                           n_filters=32,
                           kernel_shape=(7, 7),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling, 2x2
        MaxPoolingLayer(),

        # Convolutional, 5x5, 16 filters, ReLU
        ConvolutionalLayer(n_filters=16,
                           kernel_shape=(5, 5),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling 2x2
        MaxPoolingLayer(),

        # Convolutional, 3x3, 8 filters, ReLU
        ConvolutionalLayer(n_filters=16,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),

        # Max Pooling 2x2
        MaxPoolingLayer(),

        # Flattening to (1, n)
        FlatteningLayer(),

        # Dense, 1024, ReLU
        DenseLayer(n_neurons=1024,
                   activation=funs.relu,
                   activation_deriv=funs.relu_prime),

        # Dropout 25%
        DropoutLayer(probability=0.25),

        # Dense, 4, SoftMax
        DenseLayer(n_neurons=4,
                   activation=funs.softmax,
                   activation_deriv=funs.softmax_prime)
    ]

    x, y = prepare_data('images/augmented', 50)
    layers_list = final_model

    cnn = Network()

    for lay in layers_list:
        cnn.add(lay)

    cnn.compile()

    cnn.summary()

    cnn.train(inputs=x,
              correct_outputs=y,
              epochs=10,
              batch_size=32,
              shuffle=True,
              validation_split=0.25)

    cnn.save_to_json('models/final.json')
    # model = cnn.load_from_json('models/test.json')
    # model.compile()
    #
    # img_test = cv2.imread('images/augmented/ModerateDemented/1dabcf64-79b8-4a2f-8e57-c3e9ee1a64cf.jpg',
    #                       cv2.IMREAD_GRAYSCALE)
    # result = model.classify(input_for_classification=img_test)
    #
    # print([round(i, 2) for i in np.squeeze(result)])
