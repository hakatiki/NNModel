import numpy as np
import Func


class Layer:

    def __init__(self, inputs, outputs, l_r, func):
        self.lam = 0.0001
        self.eps = 0.99

        self.l_r = l_r

        self.treshold = (2 / (inputs + outputs))
        
        self.weights = (2 / (inputs+outputs)) * np.random.randn(inputs, outputs)
        self.biases = (2 / (inputs + outputs)) * np.random.randn(1, outputs)
        
        self.func = func.f
        self.func_d = func.d_f

        self.input_vec = np.zeros((inputs, 1))
        self.activation = np.zeros((outputs, 1))

        self.d_prev_w = np.zeros((inputs, outputs))
        self.d_prev_b = np.zeros((1, outputs))

    def forward_pass(self, input_vec):
        self.input_vec = input_vec

        mult = np.matmul(input_vec, self.weights)
        add = mult + self.biases
        self.activation = self.func(add)
        # print(self.activation, '\n')
        return self.activation

    def backprop(self, d_prev_layer):
        d_activation = self.func_d(self.activation)
        d_act = np.multiply(d_activation, d_prev_layer)
        d_next_prev = np.matmul(d_act, self.weights.T)

        delta_biases = d_act + self.d_prev_b * self.eps
        
        delta_weights = np.matmul(self.input_vec.T, d_act) / len(d_prev_layer)
        # clip grandient at a treshold (treshold := ?)
        scaler_val = self.treshold / (np.maximum(np.abs(delta_weights), self.treshold))
        delta_weights = delta_weights * scaler_val + self.d_prev_w * self.eps
        self.d_prev_w = delta_weights

        self.biases = self.biases - np.mean(delta_biases) * self.l_r
        # update ws with gradients and l2 norm
        self.weights = self.weights - delta_weights * self.l_r - self.weights * self.lam
        # print(np.mean(delta_weights), np.mean(self.weights))
        return d_next_prev

    def Get_d_func(self):
        return self.func_d
