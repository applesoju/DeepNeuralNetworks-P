import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0) * 1

def stable_softmax(x):
    exps = np.exp(x - np.max(x))
    sums = np.sum(exps)

    return exps / sums

def softmax_prime(softmax_output):
    s = softmax_output.reshape(-1, 1)
    return np.diag(s) - s * s.T

def mse(correct_val, predicted_val):
    return np.mean(np.power(correct_val - predicted_val, 2))

def mse_prime(correct_val, predicted_val):
    return 2 * (predicted_val - correct_val) / correct_val.size

def cross_entropy(x, y):
    m = y.shape[0]
    p = stable_softmax(x)

    log_likehood = -np.log(p[range(m), y])
    loss = np.sum(log_likehood) / m

    return loss

def cross_entropy_prime(x, y):
    m = y.shape[0]

    grad = stable_softmax(x)
    grad[range(m), y] -= 1
    grad /= m

    return grad