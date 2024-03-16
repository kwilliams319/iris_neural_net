import numpy as np
from numpy.random import normal
from itertools import count
from torch.utils.tensorboard import SummaryWriter

from Utils import Utils


class NumpyModel:
    def __init__(self):
        hl_size = 256
        self.params = {'A1': normal(size=(4, hl_size), scale=.1),
                       'b1': normal(size=(1, hl_size), scale=.1),
                       'A2': normal(size=(hl_size, 3), scale=.1),
                       'b2': normal(size=(1, 3), scale=.1)}
        self.graph = None

    def parameters(self):
        return self.params

    def forward(self, x):
        A1, b1, A2, b2 = [self.params[key] for key in ['A1', 'b1', 'A2', 'b2']]

        h1i = x @ A1 + b1  # hidden layer 1 intermediate
        h1 = np.maximum(h1i, 0)  # hidden layer 1
        h2 = h1 @ A2 + b2  # hidden layer 2

        # todo implement torch.no_grad() like feature
        Graph.set(params=self.params, activations={'h1i': h1i, 'h1': h1, 'h2': h2}, inputs=x)
        return h2


class Graph:
    params = None
    activations = None
    inputs = None
    grads = None

    @staticmethod
    def set(params, activations, inputs):
        Graph.params = params
        Graph.activations = activations
        Graph.inputs = inputs


class CrossEntropyLoss:
    def __init__(self, outputs, labels):
        self.labels = labels
        # clipping to prevent float overflow, min float in denominator to prevent divide by zero
        max_float, min_float = np.finfo(np.float64).max, np.finfo(np.float64).eps
        exponentials = np.clip(np.exp(outputs), 0, max_float)
        softmax_probs = exponentials / np.clip(np.sum(exponentials, axis=1, keepdims=True) + min_float, 0, max_float)
        self.prob_targets = softmax_probs[labels.astype(bool)]
        self.loss_scalar = np.mean(-np.log(self.prob_targets))

    def item(self):
        return self.loss_scalar

    def backward(self):
        A1, b1, A2, b2 = [Graph.params[key] for key in ['A1', 'b1', 'A2', 'b2']]
        h1i, h1, h2 = [Graph.activations[key] for key in ['h1i', 'h1', 'h2']]
        x = Graph.inputs

        # partial of loss wrt (hidden layer 2) output
        dl_dh2 = np.zeros(self.labels.shape)
        dl_dh2[self.labels.astype(bool)] = -1 + self.prob_targets

        dl_dh1 = dl_dh2 @ A2.T  # partial of loss wrt hidden layer 1 output
        dl_A2 = h1.T @ dl_dh2  # partial of loss wrt linear layer 2 A2 matrix
        dl_b2 = np.ones(b2.shape)  # partial of loss wrt linear layer 2 b2 bias vector

        dl_dh1i = dl_dh1 * (h1i > 0)  # partial of loss wrt hidden layer 1 intermediate output (pre activation)
        dl_A1 = x.T @ dl_dh1i  # partial of loss wrt linear layer 1 A1 matrix
        dl_b1 = np.ones(b1.shape)  # partial of loss wrt linear layer 1 b1 bias vector

        Graph.grads = {'A1': dl_A1, 'b1': dl_b1, 'A2': dl_A2, 'b2': dl_b2}


class SGD:
    def __init__(self, parameters, lr, momentum, weight_decay=.1):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.grads = None
        self.weight_decay = weight_decay

    def step(self):
        if self.grads is None:
            self.grads = Graph.grads
        else:
            for key in self.grads.keys():
                self.grads[key] = self.momentum * self.grads[key] + (1 - self.momentum) * Graph.grads[key]

        for key in self.parameters.keys():
            self.parameters[key] -= self.grads[key] * self.lr

        if self.weight_decay:
            self.parameters[key] *= (1 - self.lr*self.weight_decay)


class IrisChallenge:
    def __init__(self):
        self.data = Utils.split_data(Utils.load_data(), numpy=True)
        self.model = NumpyModel()

        self.lossFunction = CrossEntropyLoss
        self.optimizer = SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.writer = SummaryWriter(log_dir='runs/Numpy')

    def train(self):
        for epoch in count(0):
            for batch in range(3):
                #self.optimizer.zero_grad()
                train_outputs = self.model.forward(self.data['train_inputs'])
                train_loss = self.lossFunction(train_outputs, self.data['train_labels'])
                train_loss.backward()
                self.optimizer.step()

            Utils.log_metrics(self.data, self.model, self.lossFunction, self.writer, epoch)


if __name__ == "__main__":
    iris = IrisChallenge()
    iris.train()

