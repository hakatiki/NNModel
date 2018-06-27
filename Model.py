import numpy as np
import matplotlib.pyplot as plt
import time


batchsize = 1
l_r = 0.01
eps = 0.9


class Model:
    
    def __init__(self, neurons):
        self.weight1 = np.random.randn(neurons, 1)
        self.bias1 = np.random.randn(neurons, 1)
 
        self.weight2 = np.random.randn( 1, neurons)
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

        loss = np.mean(np.square(add2 - y))

        m_d_loss = np.mean(add2 - y)
        # print(add2 , y)
        m_activation1 = np.mean(activation1,axis=1)
        d_bias2 = m_d_loss
        d_weight2 = m_d_loss * m_activation1
        d_activation1 = m_d_loss * self.weight2

        d_add1 = np.transpose(np.greater(m_activation1, 0).astype(int))
        d_weight1 = d_activation1 * d_add1 * np.mean(x)
        d_bias1 = d_activation1 * d_add1

        delta_bias2 = d_bias2 * l_r + self.d_bias2_prev * eps
        # print(delta_bias2.shape)
        self.bias2 = self.bias2 - delta_bias2
        self.d_bias2_prev = delta_bias2
        
        delta_bias1 = np.transpose(d_bias1) * l_r + self.d_bias1_prev * eps
        # print(delta_bias1.shape)
        self.bias1 = self.bias1 - delta_bias1
        self.d_bias1_prev = delta_bias1
        
        delta_weight1 = np.transpose(d_weight1) * l_r + self.d_weight1_prev * eps
        # print(delta_weight1.shape)
        self.weight1 = self.weight1 - delta_weight1
        self.d_weight1_prev = delta_weight1
        
        delta_weight2 = np.transpose(d_weight2) * l_r + self.d_weight2_prev * eps
        # print(delta_weight2.shape)
        self.weight2 = self.weight2 - delta_weight2
        self.d_weight2_prev = delta_weight2
        
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
        for i in range(0, len(train_data), batchsize):
            loss += model.train(train_data[i:i+batchsize], label_data[i:i+batchsize])
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