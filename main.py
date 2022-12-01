def main():
    print('Hello Deep Neural Networks')


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
