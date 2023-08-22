
"""This module contains the abstract class for all MetisFL Learners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from ..proto import learner_pb2, model_pb2
from ..utils.logger import MetisLogger


class Learner(ABC):
    """Abstract class for all MetisFL Learners. All Learners should inherit from this class."""

    @abstractmethod
    def get_weights(self) -> model_pb2.Model:
        """Returns the weights of the model."""
        return model_pb2.Model()

    @abstractmethod
    def set_weights(self, model: model_pb2.Model) -> bool:
        """Sets the weights of the given model."""
        return False

    @abstractmethod
    def train(
        self,
        model: model_pb2.Model,
        params: learner_pb2.TrainParams
    ) -> Tuple[model_pb2.Model, Dict[str, Any]]:
        """Trains the given model using the given training parameters."""
        return model_pb2.Model(), {}

    @abstractmethod
    def evaluate(
        self,
        model: model_pb2.Model,
        params: learner_pb2.EvaluationParams
    ) -> Dict[str, Any]:
        """Evaluates the given model using the given evaluation parameters."""
        return {}


def has_get_weights(learner: Learner) -> bool:
    """Returns True if the given learner has a get_weights method, False otherwise."""
    return hasattr(learner, 'get_weights')


def has_set_weights(learner: Learner) -> bool:
    """Returns True if the given learner has a set_weights method, False otherwise."""
    return hasattr(learner, 'set_weights')


def has_train(learner: Learner) -> bool:
    """Returns True if the given learner has a train method, False otherwise."""
    return hasattr(learner, 'train')


def has_evaluate(learner: Learner) -> bool:
    """Returns True if the given learner has an evaluate method, False otherwise."""
    return hasattr(learner, 'evaluate')


def has_all(learner: Learner) -> bool:
    """Returns True if the given learner has all methods, False otherwise."""
    return has_get_weights(learner) and has_set_weights(learner) and \
        has_train(learner) and has_evaluate(learner)


def try_call_get_weights(learner: Learner) -> model_pb2.Model:
    """Calls the get_weights method of the given learner if it exists, otherwise returns None."""
    if has_get_weights(learner):
        return learner.get_weights()
    return None


def try_call_set_weights(
    learner: Learner,
    model: model_pb2.Model
) -> bool:
    """Calls the set_weights method of the given learner if it exists, otherwise returns False."""
    if has_set_weights(learner):
        status = learner.set_weights(model)
        if status:
            MetisLogger.info("Applied incoming weights")
        else:
            MetisLogger.info("Failed to apply incoming weights")
        return status
    return False


def try_call_train(
    learner: Learner,
    model: model_pb2.Model,
    params: learner_pb2.TrainParams
) -> Tuple[model_pb2.Model, Dict[str, Any]]:
    """Calls the train method of the given learner if it exists, otherwise returns False."""
    if has_train(learner):
        return learner.train(model, params)
    return False


def try_call_evaluate(
    learner: Learner,
    model: model_pb2.Model,
    params: learner_pb2.EvaluationParams
) -> Dict[str, Any]:
    """Calls the evaluate method of the given learner if it exists, otherwise returns None."""
    if has_evaluate(learner):
        return learner.evaluate(model, params)
    return None
