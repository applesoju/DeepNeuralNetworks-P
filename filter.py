import numpy as np


class Filter:
    def __init__(self, h, v, values):
        self.shape = (h, v)
        self.kernel = np.array(values).reshape(self.shape)
