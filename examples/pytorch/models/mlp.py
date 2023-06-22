import torch

from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import BCELoss
from torch.optim import SGD
from typing import Dict

from numpy import vstack
from sklearn.metrics import accuracy_score


class MLP(torch.nn.Module):

    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X

    def fit(self, dataset, epochs, *args, **kwargs) -> Dict:
        # define the optimization
        criterion = BCELoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(epochs):
            print("MLP Epoch: ", epoch, flush=True)
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(dataset):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
        return {}

    def evaluate(self, dataset, *args, **kwargs) -> Dict:
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(dataset):
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            yhat = yhat.round()
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return {"accuracy": acc}
