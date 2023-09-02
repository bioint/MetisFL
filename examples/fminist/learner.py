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
        model.fit(x_train[0:20000], y_train[0:20000], epochs=5,
                  batch_size=64)
        return model.get_weights(), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test[0:3000], y_test[0:3000])
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
        model.fit(x_train[20000:40000], y_train[20000:40000], epochs=5,
                  batch_size=64)
        return model.get_weights(), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test[3000:6000], y_test[3000:6000])
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
        model.fit(x_train[40000:60000], y_train[40000:60000], epochs=5,
                  batch_size=64)
        return model.get_weights(), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test[6000:9000], y_test[6000:9000])
        print("loss: {}, accuracy: {}".format(loss, accuracy))
        return {"accuracy": float(accuracy)}


def run(server_params, learner):
    app(
        learner=learner,
        server_params=server_params,
        client_params=controller_params,
        num_training_examples=len(x_train),
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
