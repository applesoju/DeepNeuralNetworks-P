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
from network import Network, load_from_json


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
        # Convolutional, 3x3, 8 fiters, ReLU
        ConvolutionalLayer(input_shape=(200, 200),
                           n_filters=8,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),
        # Max Pooling, 2x2
        MaxPoolingLayer(),
        # Convolutional, 3x3, 12 filters, ReLU
        ConvolutionalLayer(n_filters=12,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),
        # Max Pooling 2x2
        MaxPoolingLayer(),
        # Convolutional, 3x3, 16 filters, ReLU
        ConvolutionalLayer(n_filters=16,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),
        # Max Pooling 2x2
        MaxPoolingLayer(),
        # Flattening to (1, n)
        FlatteningLayer(),
        # Dense, 512, ReLU
        DenseLayer(n_neurons=512,
                   activation=funs.relu,
                   activation_deriv=funs.relu_prime),
        # Dropout 25%
        DropoutLayer(probability=0.25),
        # Dense, 4, SoftMax
        DenseLayer(n_neurons=4,
                   activation=funs.softmax,
                   activation_deriv=funs.softmax_prime)
    ]

    simple_test = [
        ConvolutionalLayer(input_shape=(200, 200),
                           n_filters=1,
                           kernel_shape=(3, 3),
                           activation=funs.relu,
                           activation_deriv=funs.relu_prime),
        MaxPoolingLayer(),
        FlatteningLayer(),
        DenseLayer(n_neurons=512,
                   activation=funs.relu,
                   activation_deriv=funs.relu_prime),
        # DropoutLayer(probability=0.25),
        DenseLayer(n_neurons=4,
                   activation=funs.softmax,
                   activation_deriv=funs.softmax_prime),
    ]
########################################################################################################################
    # Model creation, training and saving

    x, y = prepare_data('images/augmented', 1024)
    layers_list = final_model

    cnn = Network()

    for lay in layers_list:
        cnn.add(lay)

    cnn.compile()
    cnn.summary()

    cnn.train(inputs=x,
              correct_outputs=y,
              epochs=2,
              batch_size=8,
              shuffle=False,
              validation_split=0.25)

    cnn.save_to_json('models/model_001.json')

########################################################################################################################
    # Training of loaded model example

    # cnn = load_from_json('models/1024_e12.json')
    #
    # cnn.compile()
    #
    # cnn.train(inputs=x,
    #           correct_outputs=y,
    #           epochs=3,
    #           batch_size=64,
    #           shuffle=False,
    #           validation_split=0.25)
    #
    # cnn.save_to_json('models/1024_e15.json')

########################################################################################################################
    # Classification example

    # cnn = load_from_json('models/model_001.json')
    #
    # test_imgs = [
    #     cv2.imread('images/augmented/MildDemented/ff951dd6-f361-41d0-b6c4-2de07ab87490.jpg', cv2.IMREAD_GRAYSCALE),
    #     cv2.imread('images/augmented/ModerateDemented/f41afea6-1e7c-4a4b-b7d2-5eb170fa43b4.jpg', cv2.IMREAD_GRAYSCALE),
    #     cv2.imread('images/augmented/NonDemented/db3edf65-9f53-4662-90c1-54765ec0d0c1.jpg', cv2.IMREAD_GRAYSCALE),
    #     cv2.imread('images/augmented/VeryMildDemented/e93ac360-2788-41e2-bfd3-420bdb8654e4.jpg', cv2.IMREAD_GRAYSCALE)
    # ]
    #
    # result = cnn.classify(input_for_classification=test_imgs)
    #
    # print([np.round(i, 2) for i in np.squeeze(result)])
