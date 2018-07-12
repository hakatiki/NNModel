import numpy as np


class relu:
    def f(x):
        return np.maximum(0, x)

    def d_f(x):
        return np.greater(x, 0).astype(int)


class sigm:
    def f(x):
        exp = np.exp(- x)
        sum_ = np.sum(np.exp(- x), axis=1) + 0.0001
        return np.divide(exp.T, sum_).T
    def d_f(x):
        return x*(1-x)


class identiti:
    def f(x):
        return x
    def d_f(x):
        return 1

class cross_entropy_loss:
    def loss(x, y):
        x_ = sigm.f(x)
        loss = -np.sum(y * np.log(x_))
        return loss
    def d_loss(x, y):
        x_ = sigm.f(x)
        d_loss = y - x_
        return d_loss

class squared_loss:
    def loss(x, y):
        loss = np.square(x - y)
        return loss
    def d_loss(x, y):
        d_loss = x - y
        return d_loss