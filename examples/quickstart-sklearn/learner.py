import argparse
from typing import Tuple
import warnings

import numpy as np

from controller import controller_params
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metisfl.common.types import ClientParams, ServerParams
from metisfl.common.utils import iid_partition
from metisfl.learner import Learner, app


def load_data(num_clients: int = 3) -> Tuple:
    x_train, y_train = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    y_train = y_train.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        x_train, y_train, train_size=12000, test_size=3000, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    x_chunks, y_chunks = iid_partition(
        x_train=X_train, y_train=y_train, num_partitions=num_clients)

    return x_chunks, y_chunks, X_test, y_test


model = LogisticRegression(
    penalty="l2",
    max_iter=5,
    warm_start=True,
)

# Initialize the model
n_classes = 10
n_features = 784
model.coef_ = np.zeros((n_classes, n_features))
model.classes_ = np.arange(n_classes)
if model.fit_intercept:
    model.intercept_ = np.zeros(n_classes)


class TFLearner(Learner):

    """A simple sklearn Logistic Regression learner."""

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_weights(self):
        """Returns the weights of the Logistic Regression model."""
        if model.fit_intercept:
            return [
                model.coef_,
                model.intercept_,
            ]
        return [model.coef_]

    def set_weights(self, parameters):
        """Sets the weights of the Logistic Regression model."""
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]

    def train(self, parameters, config):
        """Trains the Logistic Regression model."""

        self.set_weights(parameters)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                self.x_train,
                self.y_train,
            )
            score = model.score(self.x_test, self.y_test)
            loss = log_loss(self.y_test, model.predict_proba(self.x_test))
            print("Training: accuracy = %f, loss = %f" % (score, loss))
            
        return self.get_weights(), {"accuracy": score, "loss": loss}, {"num_train_examples": len(self.x_train)}

    def evaluate(self, parameters, config):
        """Evaluates the Logistic Regression model."""

        self.set_weights(parameters)
        score = model.score(self.x_test, self.y_test)
        loss = log_loss(self.y_test, model.predict_proba(self.x_test))
        print("Evaluation: accuracy = %f, loss = %f" % (score, loss))
        
        return {"accuracy": score, "loss": loss}


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

    x_chunks, y_chunks, x_test, y_test = load_data()

    # Setup the Learner and the server parameters based on the given index
    learner = TFLearner(x_chunks[index], y_chunks[index], x_test, y_test)
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
