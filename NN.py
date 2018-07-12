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
        loss = 0
        output = Func.sigm.f(output)
        # print('out',output)
        # print('label',y)
        loss = -np.sum(y * np.log(output))
        d_loss = y - output
        # d_loss = output - y
        # print(loss)
        for j in reversed(self.layers):
            d_loss = j.backprop(d_loss)

        return np.sum(loss)
        
    def forward_pass(self, x):
        output = x
        for i in self.layers:
            output = i.forward_pass(output)
        return output


def Train_Model():
    model = NNModel(0.0001)
    # 28 * 28 == 784
    model.add_layer(28 * 28, 10, Func.relu)
    model.add_layer(10, 10, Func.relu)
    model.add_layer(10, 10, Func.relu)
    model.add_layer(10, 2, Func.identiti)

    epoch = 1000
    examples = 1000
    
    batch = 10
    x, y = LoadData.load_next_batch(examples, 0)
    x_test, y_test = LoadData.load_next_batch(examples, examples)
    for i in range(epoch):
        train_loss = 0
        for j in range(0, examples-batch, batch):
            # print(x.shape)
            train_loss += model.train(x[j:j+batch], y[j:j+batch])
        print('Train loss is at:', train_loss)
        loss = 0
        correct_guesses = 0
        for j in range(0, examples, 1):
            x_, y_ = x_test[j], y_test[j]
            output = model.forward_pass(x_)
            output = Func.sigm.f(output)
            # print(output)
            # print(y_)
            # print()
            loss += -np.sum(y_ * np.log(output))
            if np.argmax(y_) == np.argmax(output):
                correct_guesses += 1
        print('Loss after', i + 1, ':', loss)
        print('Correct guesses after ', i + 1, ':', correct_guesses / examples)



# Train_Model()