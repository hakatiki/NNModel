import numpy as np
import matplotlib.pyplot as plt
import time


batchsize = 2
l_r = 0.001
eps = 0.9


class Model:
    
    def __init__(self, neurons):
        self.weight1 = np.random.randn(neurons, 1)
        self.bias1 = np.random.randn(neurons, 1)
 
        self.weight2 = np.random.randn(1, neurons)
        self.bias2 = np.random.randn(1)

        self.d_weight1_prev = np.zeros((neurons, 1))
        self.d_weight2_prev = np.zeros((1, neurons))
        self.d_bias1_prev = np.zeros((neurons, 1))
        self.d_bias2_prev = np.zeros(1)

    def forward_pass(self, x):
        mult1 = x * self.weight1
        add1 = mult1 + self.bias1
        activation1 = np.maximum(0, add1)
    
        mult2 = np.matmul(self.weight2, activation1)
        add2 = mult2 + self.bias2
        return add2

    def train(self, x, y):
        mult1 = x * self.weight1
        add1 = mult1 + self.bias1
        activation1 = np.maximum(0, add1)
    
        mult2 = np.matmul(self.weight2, activation1)
        add2 = mult2 + self.bias2

        loss = np.square(add2 - y)
        # print loss
        d_loss = add2 - y

        d_bias2 = d_loss
        d_weight2 = d_loss * activation1
        d_activation1 = d_loss * self.weight2

        d_add1 = np.transpose(np.greater(activation1, 0).astype(int))
        d_weight1 = d_activation1 * d_add1 * x
        d_bias1 = d_activation1 * d_add1

        self.bias2 = self.bias2 - d_bias2 * l_r - self.d_bias2_prev * eps
        self.d_bias2_prev = d_bias2 * l_r + self.d_bias2_prev * eps
        
        self.bias1 = self.bias1 - np.transpose(d_bias1) * l_r - self.d_bias1_prev * eps
        self.d_bias1_prev = np.transpose(d_bias1) * l_r + self.d_bias1_prev * eps
        
        self.weight1 = self.weight1 - np.transpose(d_weight1) * l_r - self.d_weight1_prev * eps
        self.d_weight1_prev=np.transpose(d_weight1) * l_r + self.d_weight1_prev * eps
        
        self.weight2 = self.weight2 - np.transpose(d_weight2) * l_r - self.d_weight2_prev * eps
        self.d_weight2_prev = np.transpose(d_weight2) * l_r + self.d_weight2_prev * eps
        
        return loss

    def print_w(self):
        print(self.weight1)


def Train():
    model = Model(100)
    valid = np.arange(0, 1, 0.1)
    valid_sq = np.power(valid, 0.5)
    
    train_data = np.random.rand(100)
    label_data = np.power(train_data, 0.5)
    for j in range(1000):
        loss = 0
        for i in range(len(train_data)):
            loss += model.train(train_data[i], label_data[i])
        print(loss)
        preds2 = []
        for i in valid:
            preds2.extend(model.forward_pass(i)[0])
        plt.clf()
        plt.plot(valid, valid_sq)
        plt.plot(valid, preds2)
        plt.pause(0.5)
       
    plt.plot(valid, valid_sq)  
    plt.pause(10) 


Train()