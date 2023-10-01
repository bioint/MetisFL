import argparse
from collections import OrderedDict
import numpy as np
import torch
from controller import controller_params
from data import load_data
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
import evaluate

from metisfl.common.types import ClientParams, ServerParams
from metisfl.learner import Learner, app

DEVICE = "cpu"
MODEL_URL = "distilbert-base-uncased"


def get_model_and_tokenizer():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_URL,
        num_labels=2,
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_URL)

    return model, tokenizer


class TransformerLearner(Learner):
    """A simple Transformer Learner."""

    def __init__(
        self,
        trainloader: torch.utils.data.dataloader.DataLoader,
        testloader: torch.utils.data.dataloader.DataLoader
    ):
        super().__init__()
        self.trainloader = trainloader
        self.testloader = testloader
        self.model, self.tokenizer = get_model_and_tokenizer()


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
        lr = config["learning_rate"] if "learning_rate" in config else 2e-5
        weight_decay = config["weight_decay"] if "weight_decay" in config else 0.01

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.train()
        losses = []
        accs = []
        for epoch in range(epochs):
            for batch in self.trainloader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                predicted = torch.argmax(outputs.logits, dim=-1)
                labels = batch["labels"]
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                accuracy = correct / total

                losses.append(loss.item())
                accs.append(accuracy)
            print(f"Epoch {epoch+1}: Loss {np.mean(losses)} Accuracy {np.mean(accs)}")

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

        metric = evaluate.load("accuracy")
        loss = 0
        self.model.eval()
        for batch in self.testloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            loss += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            metric.add_batch(predictions=predictions, references=labels)
        loss /= len(self.testloader.dataset)
        accuracy = metric.compute()["accuracy"]

        print(f'Evaluation accuracy: {accuracy} and loss: {loss}')
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

    trainloader, testloader = load_data()

    # Create the learner
    learner = TransformerLearner(trainloader, testloader)

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