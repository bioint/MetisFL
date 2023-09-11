import argparse

from controller import controller_params
from data import load_data
from model import get_model

from metisfl.common.types import ClientParams, ServerParams
from metisfl.learner import Learner, app


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
        return self.model.set_weights(parameters)

    def train(self, parameters, config):
        self.model.set_weights(parameters)
        batch_size = config["batch_size"] if "batch_size" in config else 64
        epochs = config["epochs"] if "epochs" in config else 3
        res = self.model.fit(x=self.x_train, y=self.y_train,
                             batch_size=batch_size, epochs=epochs)
        metadata = {
            "num_training_examples": len(self.x_train),
        }
        return self.model.get_weights(), res.history, metadata

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
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

    x_chunks, y_chunks, x_test, y_test = load_data(max_learners=max_learners)

    # Setup the Learner and the server parameters based on the given index
    learner = TFLearner(x_chunks[index], y_chunks[index], x_test, y_test)
    server_params = get_learner_server_params(index, max_learners)

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
