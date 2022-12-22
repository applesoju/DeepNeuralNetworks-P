import numpy as np

def relu(x):
    out = max(0, x)
    return out

def relu_prime():
    out = (x > 0) * 1
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
