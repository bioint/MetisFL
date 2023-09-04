import argparse
import os
from typing import Tuple

import tensorflow as tf
from controller import controller_params
from model import get_model

from metisfl.common.types import ClientParams, ServerParams, LearnerConfig
from metisfl.common.utils import iid_partition
from metisfl.learner import app
from metisfl.learner import Learner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def load_data(rescale_reshape=True) -> Tuple:
    """A helper function to load the Fashion MNIST dataset.

    Args:
        rescale_reshape (bool, optional): Whether to rescale and reshape. (Defaults to True)

    Returns:
        Tuple: A tuple containing the training and test data.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape the data
    if rescale_reshape:
        x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
        x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

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

    def train(self, parameters, config):
        self.model.set_weights(parameters)
        batch_size = config["batch_size"] if "batch_size" in config else 64
        epochs = config["epochs"] if "epochs" in config else 3
        res = self.model.fit(x=self.x_train, y=self.y_train,
                             batch_size=batch_size, epochs=epochs)
        parameters = self.model.get_weights()
        return parameters, res.history  # FIXME: check protos

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
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

    x_train, y_train, x_test, y_test = load_data()

    # Partition the data into 3 clients, iid
    x_client, y_client = iid_partition(x_train, y_train, max_learners)

    # Setup the Learner and the server parameters based on the given index
    learner = TFLearner(x_client[index], y_client[index], x_test, y_test)
    server_params = get_learner_server_params(index)

    # Setup the client parameters based on the controller parameters
    client_params = ClientParams(
        hostname=controller_params.hostname,
        port=controller_params.port,
        root_certificate=controller_params.root_certificate,
    )

    config = LearnerConfig(
        batch_size=8192,
        scaling_factor_bits=40,
        crypto_context="/home/panoskyriakis/metisfl/crypto_context.txt",
        public_key="/home/panoskyriakis/metisfl/public_key.txt",
        private_key="/home/panoskyriakis/metisfl/private_key.txt",
    )

    # Start the app
    app(
        learner=learner,
        server_params=server_params,
        client_params=client_params,
        learner_config=config,
        num_training_examples=len(x_client[index]),
    )
