import numpy as np


def linear(x):
    return x


def linear_prime(x):
    return np.ones(x.shape)


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0) * 1


def softmax(x):
    exps = np.exp(x - np.max(x))
    sums = np.sum(exps)
    out = exps / sums
    return out


def softmax_prime(x):
    soft = softmax(x)
    out = soft * (1 - soft)
    return out


def sigmoid(x):
    ex = np.e ** x
    out = ex / (ex + 1)
    return out


def sigmoid_prime(x):
    ex = np.e ** x
    f = ex / (ex + 1)
    out = f * (1 - f)
    return out


def mse(correct_val, predicted_val):
    return np.power(correct_val - predicted_val, 2)


def mse_prime(correct_val, predicted_val):
    return 2 * (predicted_val - correct_val) / correct_val.size