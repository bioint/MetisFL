import argparse
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from controller import controller_params
from model import Model
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from metisfl.common.types import ClientParams, ServerParams
from metisfl.common.utils import iid_partition
from metisfl.learner import app
from metisfl.learner import Learner

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(num_learners: int) -> Tuple:
    """Load CIFAR-10  and partition it into num_learners clients, iid."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    x_chunks, y_chunks = iid_partition(
        x_train=trainset.data, y_train=trainset.targets, num_partitions=num_learners)

    # Convert the numpy arrays to torch tensors and make it channels first
    x_chunks = [torch.Tensor(x).permute(0, 3, 1, 2) for x in x_chunks]
    y_chunks = [torch.Tensor(y).long() for y in y_chunks]
    trainset_chunks = [TensorDataset(x, y) for x, y in zip(x_chunks, y_chunks)]

    # Same for the test set
    test_data = torch.Tensor(testset.data).permute(0, 3, 1, 2)
    test_labels = torch.Tensor(testset.targets).long()
    testset = TensorDataset(test_data, test_labels)

    return trainset_chunks, testset


class TorchLearner(Learner):

    """A simple PyTorch Learner."""

    def __init__(
        self,
        trainset: TensorDataset,
        testset: TensorDataset,
    ):
        super().__init__()
        self.trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model = Model().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_weights(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, parameters):
        params = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(v.copy()) for k, v in params})
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, parameters, config):
        self.set_weights(parameters)
        epochs = config["epochs"] if "epochs" in config else 1
        losses = []
        accs = []
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                accuracy = correct / total

                losses.append(loss.item())
                accs.append(accuracy)

        metrics = {
            "accuracy": np.mean(accs),
            "loss": np.mean(losses),
        }
        metadata = {
            "num_training_examples": len(self.trainloader.dataset),
        }
        return self.get_weights(), metrics, metadata

    def evaluate(self, parameters, config):
        self.set_weights(parameters)

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / total

        return {"accuracy": float(accuracy), "loss": float(loss)}


def get_learner_server_params(learner_index, max_learners=3):
    """A helper function to get the server parameters for a learner. """
    ports = list(range(50002, 50002 + max_learners))
    return ServerParams(
        hostname="localhost",
        port=ports[learner_index],
    )


if __name__ == "__main__":
    """The main function. It loads the data, creates a learner, and starts the learner server."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner")
    parser.add_argument("-m", "--max-learners", type=int, default=3)

    args = parser.parse_args()
    index = int(args.learner) - 1
    max_learners = args.max_learners

    # Partition the data into 3 clients, iid
    trainset_chuncks, testset = load_data(num_learners=max_learners)

    # Create the learner
    learner = TorchLearner(trainset_chuncks[index], testset)

    # Setup the client parameters
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
    )

    # Setup the server parameters of the learner
    server_params = get_learner_server_params(index)

    # Start the app
    app(
        learner=learner,
        server_params=server_params,
        client_params=client_params,
    )
