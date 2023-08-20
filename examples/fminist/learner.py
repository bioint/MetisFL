import argparse

from dataset import load_data
from model import get_model

from metisfl.learner.app import app
from metisfl.learner.learner import Learner
from metisfl.proto import model_pb2
from env import controller_params, learner_1, learner_2

model = get_model()
x_train, y_train, x_test, y_test = load_data()


class MyLearner(Learner):
    def get_weights(self):
        return model_pb2.Model()

    def set_weights(self, parameters):
        return True

    def train(self, parameters, config):
        print("Training")
        print(parameters)
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}


def run(server_params):
    app(
        learner=MyLearner(),
        server_params=server_params,
        client_params=controller_params,
        num_training_examples=len(x_train),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner")

    args = parser.parse_args()
    if args.learner == "1":
        run(learner_1)
    elif args.learner == "2":
        run(learner_2)
