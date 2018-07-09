import numpy as np
import Func

class Layer:

    def __init__(self, inputs, outputs, l_r, func):
        self.inp_size = inputs
        self.out_size = outputs
        self.l_r = l_r
        self.weights = (2 / inputs) * np.random.randn(inputs, outputs)
        self.biases = (2 / inputs) * np.random.randn(1, outputs)
        self.func = func.f
        self.func_d = func.d_f

        self.input_vec = np.zeros((inputs, 1))
        self.activation = np.zeros((outputs, 1))

        self.d_weights = np.zeros((inputs, outputs))
        self.d_biases = np.zeros((outputs, 1))

    def forward_pass(self, input_vec):
        self.input_vec = np.mean(input_vec, axis=0).reshape(1, -1)

        mult = np.matmul(input_vec, self.weights)
        add = mult + self.biases
        self.activation = self.func(add)

        return self.activation

    def backprop(self, d_prev_layer):
        self.activation = np.mean(self.activation, axis=0) 
        d_act = np.multiply(self.func_d(self.activation), d_prev_layer).reshape((self.out_size, 1))

        delta_biases = self.l_r * d_act.T
        self.biases = self.biases - delta_biases
        # Faszom se tudja hogy jó-e, de valószínüleg oké 
        delta_weights = self.l_r * np.matmul(d_act, self.input_vec).T
        # print('Mean change in weights is:', np.mean(delta_weights))
        self.weights = self.weights - delta_weights
        # menő ez is
        d_next_prev = np.matmul(d_act.T, self.weights.T)
        return d_next_prev

    def Get_d_func(self):
        return self.d_func
