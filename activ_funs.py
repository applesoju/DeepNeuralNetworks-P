import numpy as np


def linear(x):
    return x


def relu(x):
    out = max(0, x)
    return out


def sigmoid(x):
    ex = np.e ** x
    out = ex / (ex + 1)
    return out
