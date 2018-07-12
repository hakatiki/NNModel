import numpy as np
import matplotlib.pyplot as plt
import time

from NN import NNModel
import Func


l_r = 0.001
eps = 0.9
lam = 0.0001


class Model:
    
    def __init__(self, neurons):
        self.weight1 = np.random.randn(neurons, 1)
        self.bias1 =  np.random.randn(neurons, 1)
 
        self.weight2 =  np.random.randn(1, neurons)
        self.bias2 =  np.random.randn(1)

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

        loss = np.mean(np.square(add2 - y))

        m_d_loss = np.mean(add2 - y)
        # print(add2 , y)
        m_activation1 = np.mean(activation1, axis=1)
        d_bias2 = m_d_loss
        d_weight2 = m_d_loss * m_activation1
        d_activation1 = m_d_loss * self.weight2

        d_add1 = np.transpose(np.greater(m_activation1, 0).astype(int))
        d_weight1 = d_activation1 * d_add1 * np.mean(x)
        d_bias1 = d_activation1 * d_add1
 
        delta_bias2 = d_bias2 * l_r + self.d_bias2_prev * eps
        # print(delta_bias2.shape)
        self.bias2 = self.bias2 - delta_bias2 - lam * self.bias2
        self.d_bias2_prev = delta_bias2
        
        delta_bias1 = np.transpose(d_bias1) * l_r + self.d_bias1_prev * eps
        # print(delta_bias1.shape)
        self.bias1 = self.bias1 - delta_bias1 - lam * self.bias1
        self.d_bias1_prev = delta_bias1
        
        delta_weight1 = np.transpose(d_weight1) * l_r + self.d_weight1_prev * eps
        # print(delta_weight1.shape)
        self.weight1 = self.weight1 - delta_weight1 - lam * self.weight1
        self.d_weight1_prev = delta_weight1
        
        delta_weight2 = np.transpose(d_weight2) * l_r + self.d_weight2_prev * eps
        # print(delta_weight2.shape)
        self.weight2 = self.weight2 - delta_weight2 - lam * self.weight2 
        self.d_weight2_prev = delta_weight2
        
        return loss

    def print_w(self):
        print("Weights means are:")
        print(np.mean(np.abs(self.weight1)))
        print(np.mean(np.abs(self.weight2)))
        print(np.mean(np.abs(self.bias1)))
        print(np.mean(np.abs(self.bias2)))


def Train():
    model = NNModel(0.01)
    model.add_layer(1, 11, Func.relu)
    model.add_layer(11, 10, Func.relu)
    model.add_layer(10, 1, Func.identiti)
    batchsize = 1
    valid = np.arange(0, 1, 0.01)
    valid_sq = np.power(valid, 2)
    
    train_data = np.random.rand(100)
    label_data = np.power(train_data, 2)
    
    for j in range(200):
        loss = 0
        for i in range(0, len(train_data), batchsize):
            t_d = train_data[i:i + batchsize].reshape((-1, 1))
            t_l = label_data[i:i + batchsize].reshape((-1, 1))
            loss += model.train(t_d, t_l)
        # print(loss)
        preds2 = model.forward_pass(valid.reshape((-1, 1)))
        # print(preds2.reshape((1,-1)))
        plt.clf()
        plt.plot(valid, valid_sq)
        plt.plot(valid, preds2)
        plt.pause(0.5)
    plt.plot(valid, valid_sq)  
    plt.pause(10) 


Train()