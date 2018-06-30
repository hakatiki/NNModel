import numpy as np



class Layer:

    def __init__(self, inputs, outputs, func):

        self.weights = (1 / inputs) * np.random.randn(inputs, outputs)
        self.bias = (1 / inputs) * np.random.randn(outputs, 1)
        self.func = func

        self.input_vec = np.zeros((inputs, 1))
        self.activation = np.zeros((outputs, 1))

        self.d_weight1_prev = np.zeros((inputs, outputs))
        self.d_bias1_prev = np.zeros((outputs, 1))

    def forward_pass(self, input_vec):
        self.input_vec = input_vec

        mult = np.matmul(input_vec, self.weights)
        add = mult + self.bias
        self.activation = self.func(add)

        return self.activation
