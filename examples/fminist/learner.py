import argparse

from dataset import load_data
from examples.fminist.dataset_keras import partition_data_iid
from model import get_model

from metisfl.learner.app import app
from metisfl.learner.learner import Learner
from env import controller_params, learner_1, learner_2, learner_3


model = get_model()
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
x_train, y_train, x_test, y_test = load_data()

x_chunks, y_chunks = partition_data_iid(x_train, y_train, 3)


class MyLearner1(Learner):
    def get_weights(self):
        return model.get_weights()

    def set_weights(self, parameters):
        model.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model.set_weights(parameters)
        print(parameters[0])
        model.fit(x_chunks[0], y_chunks[0], epochs=3,
                  batch_size=64)
        parameters = model.get_weights()
        print(parameters[0])
        return parameters, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


class MyLearner2(Learner):
    def get_weights(self):
        return model.get_weights()

    def set_weights(self, parameters):
        model.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model.set_weights(parameters)
        print(parameters[0])
        model.fit(x_chunks[1], y_chunks[1], epochs=3,
                  batch_size=64)
        parameters = model.get_weights()
        print(parameters[0])
        return parameters, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


class MyLearner3(Learner):
    def get_weights(self):
        return model.get_weights()

    def set_weights(self, parameters):
        model.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model.set_weights(parameters)
        print(parameters[0])
        model.fit(x_chunks[2], y_chunks[2], epochs=3,
                  batch_size=64)
        parameters = model.get_weights()
        print(parameters[0])
        return parameters, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


def run(server_params, learner):
    app(
        learner=learner,
        server_params=server_params,
        client_params=controller_params,
        num_training_examples=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learner")
    args = parser.parse_args()
    if args.learner == "1":
        run(learner_1, MyLearner1())
    elif args.learner == "2":
        run(learner_2, MyLearner2())
    elif args.learner == "3":
        run(learner_3, MyLearner3())
