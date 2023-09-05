import argparse
from collections import OrderedDict
from typing import Tuple

import torch
import torchvision.transforms as transforms
from controller import controller_params
from model import Model
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from metisfl.common.types import ClientParams, ServerParams
from metisfl.common.utils import iid_partition
from metisfl.learner import app
from metisfl.learner.learner import Learner

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(num_learners: int) -> Tuple:
    """Load CIFAR-10  and partition it into 3 clients, iid."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    return trainset, testset


class TorchLearner(Learner):

    """A simple PyTorch Learner."""

    def __init__(
        self,
        trainset: TensorDataset,
        testset: TensorDataset,
    ):
        super().__init__()
        self.trainset = trainset
        self.testset = testset
        self.model = Model().to(DEVICE)

    def get_weights(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, parameters):
        params = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params})
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, parameters, config):
        self.set_weights(parameters)

        batch_size = config["batch_size"] if "batch_size" in config else 64
        epochs = config["epochs"] if "epochs" in config else 3
        train_loader = DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

        return self.get_weights()

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        batch_size = config["batch_size"] if "batch_size" in config else 64
        test_loader = DataLoader(
            self.testset, batch_size=batch_size, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
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
    trainset, testset = load_data(num_learners=max_learners)
    x_chunks, y_chunks = iid_partition(
        x_train=trainset.data, y_train=trainset.targets, num_partitions=max_learners)

    # Get the data for the current learner
    trainset = TensorDataset(torch.Tensor(
        x_chunks[index]), torch.Tensor(y_chunks[index]))

    # Create the learner
    learner = TorchLearner(trainset, testset)

    server_params = get_learner_server_params(index)

    # Setup the client parameters based on the controller parameters
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
        root_certificate=controller_params.root_certificate,
    )

    # Start the app
    app(
        learner=learner,
        server_params=server_params,
        client_params=client_params,
    )

# x, y = load_data(3)
# learner = TorchLearner(x[0], y)
# print(learner.evaluate(learner.get_weights(), {}))
