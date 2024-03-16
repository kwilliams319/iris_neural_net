import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from itertools import count
from Utils import Utils


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.classifier(x)


class TorchModel:
    def __init__(self):
        self.A1 = nn.parameter.Parameter(2*torch.rand((4, 256)) - 1)
        self.b1 = nn.parameter.Parameter(2*torch.rand((1, 256)) - 1)
        self.A2 = nn.parameter.Parameter(2*torch.rand((256, 3)) - 1)
        self.b2 = nn.parameter.Parameter(2*torch.rand((1, 3)) - 1)

    def parameters(self):
        return [self.A1, self.b1, self.A2, self.b2]

    def forward(self, x):
        h1 = x @ self.A1 + self.b1
        h1 = torch.maximum(h1, torch.zeros(h1.size()))
        h2 = h1 @ self.A2 + self.b2
        return h2


class IrisChallenge:
    def __init__(self):
        self.data = Utils.split_data(Utils.load_data())

        self.model = TorchModel()
        # self.model = Model()

        self.lossFunction = nn.CrossEntropyLoss()
        # self.lossFunction = nn.MSELoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.writer = SummaryWriter(log_dir='runs/PyTorch')

    def train(self):
        for epoch in count(0):
            for batch in range(3):
                self.optimizer.zero_grad()
                train_outputs = self.model.forward(self.data['train_inputs'])
                train_loss = self.lossFunction(train_outputs, self.data['train_labels'])
                train_loss.backward()
                self.optimizer.step()
            Utils.log_metrics(self.data, self.model, self.lossFunction, self.writer, epoch)


if __name__ == "__main__":
    iris = IrisChallenge()
    iris.train()

    # softmax_probs = torch.exp(train_outputs) / (torch.sum(torch.exp(train_outputs), dim=1, keepdims=True) + np.finfo(np.float64).eps)
    # prob_targets = softmax_probs[torch.tensor(labels, dtype=torch.bool)]
    # loss = torch.mean(-torch.log(prob_targets))

