import numpy as np

def relu(x):
    out = max(0, x)
    return out

def relu_prime(x):
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

def mse(correct_val, predicted_val):
    return np.mean(np.power(correct_val - predicted_val, 2))

def mse_prime(correct_val, predicted_val):
    return 2 * (predicted_val - correct_val) / correct_val.size