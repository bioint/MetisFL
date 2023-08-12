import abc
from metisfl.models.types import ModelWeightsDescriptor


class Learner(abc.ABC):
    """Abstract class for Learner. All Learner classes should inherit from this class."""

    @abc.abstractmethod
    def get_weights(self):
        pass

    @abc.abstractmethod
    def set_weights(self, model_weights_descriptor: ModelWeightsDescriptor):
        pass

    @abc.abstractmethod
    def train(self, learning_task_pb, hyperparameters_pb):
        pass

    @abc.abstractmethod
    def evaluate(self, learning_task_pb, hyperparameters_pb):
        pass
