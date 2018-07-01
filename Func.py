import numpy as np


class relu:
    def f(x):
        return np.maximum(0, x)

    def d_f(x):
        return np.greater(x, 0).astype(int)


class sigm:
    def f(x):
        exp = np.exp( - x)
        sum_ = np.sum(np.exp( - x),axis=1) + 0.000001
        return np.divide(exp.T, sum_).T
    def d_f(x):
        return x(1-x)


class identiti:
    def f(x):
        return x
    def d_f(x):
        return 1

    