import argparse

import tensorflow as tf

from dataset import load_data
from env import learner_1, learner_2, learner_3
from model import get_model

from metisfl.common.types import ClientParams
from metisfl.common.utils import iid_partition
from metisfl.learner import app
from metisfl.learner.learner import Learner

model1 = get_model()
model1.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model2 = get_model()
model2.compile(
    loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001), metrics=["accuracy"]
)


x_train, y_train, x_test, y_test = load_data()
x_chunks, y_chunks = iid_partition(x_train, y_train, 3)


class MyLearner1(Learner):
    def get_weights(self):
        return model1.get_weights()

    def set_weights(self, parameters):
        model1.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model1.set_weights(parameters)
        model1.fit(x_chunks[0], y_chunks[0], epochs=3,
                   batch_size=64)
        parameters = model1.get_weights()
        return parameters, {}

    def evaluate(self, parameters, config):
        model1.set_weights(parameters)
        loss, accuracy = model1.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


class MyLearner2(Learner):
    def get_weights(self):
        return model2.get_weights()

    def set_weights(self, parameters):
        model2.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model2.set_weights(parameters)
        model2.fit(x_chunks[1], y_chunks[1], epochs=3,
                   batch_size=64)
        parameters = model2.get_weights()
        return parameters, {}

    def evaluate(self, parameters, config):
        model2.set_weights(parameters)
        loss, accuracy = model2.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


class MyLearner3(Learner):
    def get_weights(self):
        return model2.get_weights()

    def set_weights(self, parameters):
        model2.set_weights(parameters)
        return True

    def train(self, parameters, config):
        model2.set_weights(parameters)
        model2.fit(x_chunks[2], y_chunks[2], epochs=3,
                   batch_size=64)
        parameters = model2.get_weights()
        return parameters, {}

    def evaluate(self, parameters, config):
        model2.set_weights(parameters)
        loss, accuracy = model2.evaluate(x_test, y_test)
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


def run(server_params, learner):
    app(
        learner=learner,
        server_params=server_params,
        client_params=ClientParams(
            hostname="localhost",
            port=50051,
            root_certificate="/home/panoskyriakis/metisfl/ca-cert.pem",
        ),
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
