import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf

from metisfl.common.types import ClientParams, ServerParams
from metisfl.common.utils import iid_partition
from metisfl.learner import app
from metisfl.learner.learner import Learner

from model import get_model
from controller import controller_params


def load_data(rescale_reshape=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A helper function to load the Fashion MNIST dataset.

    Args:
        rescale_reshape (bool, optional): Whether to rescale and reshape. (Defaults to True)

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): A tuple containing the training and test data.
    """

    # Load data using TensorFlow Keras API
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize and reshape the data
    if rescale_reshape:
        x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
        x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

    # Return the data
    return x_train, y_train, x_test, y_test


class TFLearner(Learner):

    """A simple TensorFlow Learner."""

    def __init__(self, x_train, y_train, x_test, y_test):
        super().__init__()
        self.model = get_model()

        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, parameters):
        self.model.set_weights(parameters)
        return True

    def train(self, parameters, config):
        self.model.set_weights(parameters)

        batch_size = config["batch_size"] if "batch_size" in config else 64
        epochs = config["epochs"] if "epochs" in config else 3

        res = self.model.fit(x=self.x_train, y=self.y_train,
                             batch_size=batch_size, epochs=epochs)

        print(res)

        parameters = self.model.get_weights()
        return parameters, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)

        return {"accuracy": float(accuracy), "loss": float(loss)}


def get_learner_server_params(learner_index):
    """A helper function to get the server parameters for a learner. """

    ports = [50052, 50053, 50054]

    return ServerParams(
        hostname="localhost",
        port=ports[learner_index],
        root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
        private_key="/home/panoskyriakis/metisfl/server-key.pem",
        server_certificate="/home/panoskyriakis/metisfl/server-cert.pem",
    )


if __name__ == "__main__":
    """The main function. It loads the data, creates a learner, and starts the learner server."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner")

    args = parser.parse_args()
    index = int(args.learner) - 1

    x_train, y_train, x_test, y_test = load_data()

    # Partition the data into 3 clients, iid
    x_client, y_client = iid_partition(x_train, y_train, 3)

    # Setup the Learner and the server parameters based on the given index
    learner = TFLearner(x_client[index], y_client[index], x_test, y_test)
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
        num_training_examples=len(x_client[index]),
    )
