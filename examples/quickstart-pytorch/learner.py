import argparse
from collections import OrderedDict

import numpy as np
import torch
from controller import controller_params
from data import load_data
from model import Model
from torch.utils.data import DataLoader, TensorDataset

from metisfl.common.types import ClientParams, ServerParams
from metisfl.learner import Learner, app

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        epochs = config["epochs"] if "epochs" in config else 5
        losses = []
        accs = []
        for epoch in range(epochs):
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
            print(
                f"Finished epoch {epoch + 1} with loss {loss.item()} and accuracy {accuracy}")

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
        print(f"Evaluation Accuracy: {accuracy}, Evaluation Loss: {loss}")

        return {"accuracy": float(accuracy), "loss": float(loss)}


def get_learner_server_params(learner_index, max_learners):
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
    trainset_chuncks, testset = load_data(max_learners)

    # Create the learner
    learner = TorchLearner(trainset_chuncks[index], testset)

    # Setup the client parameters
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
    )

    # Setup the server parameters of the learner
    server_params = get_learner_server_params(index, max_learners)

    # Start the app
    app(
        learner=learner,
        server_params=server_params,
        client_params=client_params,
    )
