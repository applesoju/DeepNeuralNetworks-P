import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_prime(softmax_output):
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def mse(correct_val, predicted_val):
    return np.mean(np.power(correct_val - predicted_val, 2))

def mse_prime(correct_val, predicted_val):
    return 2 * (predicted_val - correct_val) / correct_val.size