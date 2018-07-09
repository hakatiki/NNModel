import numpy as np
from Layer import Layer
import Func
import LoadData


class NNModel:

    def __init__(self, l_r):
        self.l_r = l_r
        self.layers = []

    def add_layer(self, inputs, outputs, func):
        layer = Layer(inputs, outputs, self.l_r, func)
        self.layers.append(layer)

    def train(self, x, y):
        output = x
        for i in self.layers:
            output = i.forward_pass(output)
        output = Func.sigm.f(output)
        print(output)
        loss = - np.sum(y * np.log(output))
        d_loss = y - output
        d_loss = np.mean(d_loss, axis=0).reshape(1, -1)
        print(d_loss)
        for i in reversed(self.layers):
            d_loss = i.backprop(d_loss)

        return loss
        
    def forward_pass(self, x):
        output = x
        for i in self.layers:
            output = i.forward_pass(x)
        return output


def Train_Model():
    model = NNModel(0.001)
    # 28 * 28 == 784
    model.add_layer(28 * 28, 100, Func.relu)
    model.add_layer(100, 111, Func.relu)
    model.add_layer(111, 2, Func.identiti)

    epoch = 100
    examlpes = 1000
    
    batch = 4
    
    for i in range(epoch):
        train_loss = 0
        for j in range(0, examlpes, batch):
            x, y = LoadData.load_next_batch(batch, j)
            # print(x.shape)
            train_loss += model.train(x, y)

        loss = 0
        correct_guesses = 0
        for j in range(examlpes, examlpes + 1000, 1):
            x, y = LoadData.load_next_batch(1, j)
            output = model.forward_pass(x)
            loss += - np.sum(y * np.log(output))
            if np.argmax(y) == np.argmax(output):
                correct_guesses += 1
        print('Loss after', i, ':', loss)
        print('Correct guesses after ', i, ':', correct_guesses)


Train_Model()