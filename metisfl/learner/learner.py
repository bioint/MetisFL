
"""This module contains the abstract class for all MetisFL Learners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class Learner(ABC):
    """Abstract class for all MetisFL Learners. All Learners should inherit from this class."""

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        """Returns the weights of the model as a list of numpy arrays.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays representing the weights of the model.
        """
        return np.array([])

    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Sets the weights of the model using the given weights.

        Parameters
        ----------
        weights : List[np.ndarray]
            A list of numpy arrays representing the weights of the model to be set.

        """
        return False

    @abstractmethod
    def train(
        self,
        weights: List[np.ndarray],
        params: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Trains the model using the given training parameters.

        Parameters
        ----------
        weights : List[np.ndarray]
            A list of numpy arrays representing the weights of the model to be trained.
        params : Dict[str, Any]
            A dictionary of training parameters.

        Returns
        -------
        Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]
            A tuple containing the following:
                - A list of numpy arrays representing the weights of the model after training.
                - A dictionary of the metrics computed during training.
                - A dictionary of training metadata, such as the number 
                    of completed epochs, batches, processing time, etc.
        """
        return [], {}

    @abstractmethod
    def evaluate(
        self,
        model: List[np.ndarray],
        params: Dict[str, Any]
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


def try_call_get_weights(learner: Learner) -> List[np.ndarray]:
    """Calls the get_weights method of the given learner.

    Parameters
    ----------
    learner : Learner
        The Learner object to call the get_weights method on.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays representing the weights of the model.

    Raises
    ------
    ValueError
        If the learner does not have a get_weights method.
    ValueError
        If the weights returned by the get_weights method are not a list of numpy arrays.
    """

    if has_get_weights(learner):
        weights = learner.get_weights()

        if not isinstance(weights, list) or len(weights) == 0 or \
                not all(isinstance(weight, np.ndarray) for weight in weights):
            raise ValueError("Weights must be a list of numpy arrays")

        return weights

    raise ValueError("Learner does not have a get_weights method")


def try_call_set_weights(
    learner: Learner,
    weights: List[np.ndarray]
) -> None:
    """Calls the set_weights method of the given learner.

    Parameters
    ----------
    learner : Learner
        The Learner object to call the set_weights method on.
    weights : List[np.ndarray]
        The weights of the model to be set.        

    Raises
    ------
    ValueError
        If the learner does not have a set_weights method.
    """

    if has_set_weights(learner):
        return learner.set_weights(weights)

    raise ValueError("Learner does not have a set_weights method")


def try_call_train(
    learner: Learner,
    weights: List[np.ndarray],
    params: Dict[str, Any]
) -> Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Tries to call the train method of the given learner.

    Parameters
    ----------
    learner : Learner
        The Learner object to call the train method on.
    weights : List[np.ndarray]
        The weights of the model to be trained.
    params : Dict[str, Any]
        A dictionary of training parameters or metrics to be computed during training.

    Returns
    -------
    Tuple[List[np.ndarray], Dict[str, Any], Dict[str, Any]]
        A tuple containing the following:
            - A list of numpy arrays representing the weights of the model after training.
            - A dictionary of the metrics computed during training.
            - A dictionary of training metadata. 

    Raises
    ------
    ValueError
        If the learner does not have a train method.
    ValueError
        If the metrics specified in the training parameters are not found in the training results.
    """

    if has_train(learner):
        train_res = learner.train(weights, params)

        if not isinstance(train_res, tuple) or len(train_res) not in [2, 3]:
            raise ValueError(
                "Learner.train must return a tuple of length 2 or 3")

        weights = train_res[0]
        metrics = train_res[1]
        metadata = train_res[2] if len(train_res) == 3 else {}


        if not isinstance(weights, list) or len(weights) == 0 or \
                not all(isinstance(weight, np.ndarray) for weight in weights):
            raise ValueError(
                "Learner.train must return a list of numpy arrays")

        if not isinstance(metrics, dict):
            raise ValueError(
                "Learner.train must return a dictionary of metrics")
            
        if not isinstance(metadata, dict):
            raise ValueError(
                "Learner.train must return a dictionary of metadata")

        for metrics in params.get('metrics', []):
            if metrics not in train_res[1]:
                raise ValueError(
                    f"Metric '{metrics}' not found in training metadata")
                
        for required_metadata in params.get('metadata', []):
            if required_metadata not in metadata:
                raise ValueError(
                    f"Metadata '{required_metadata}' not found in training metadata")

        return weights, metrics, metadata

    raise ValueError("Learner does not have a train method")


def try_call_evaluate(
    learner: Learner,
    weights: List[np.ndarray],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Tries to call the evaluate method of the given learner.

    Parameters
    ----------
    learner : Learner
        The Learner object to call the evaluate method on.
    weights : List[np.ndarray]
        The weights of the model to be evaluated.
    params : Dict[str, Any]
        A dictionary of evaluation parameters.

    Returns
    -------
    Dict[str, Any]
        A dictionary of the metrics computed during evaluation.

    Raises
    ------
    ValueError
        If the learner does not have an evaluate method.
    ValueError
        If the metrics specified in the evaluation parameters are not found in the evaluation results.
    """

    if has_evaluate(learner):
        eval_res = learner.evaluate(weights, params)

        if not isinstance(eval_res, dict):
            raise ValueError(
                "Learner.evaluate must return a dictionary of metrics")

        for metrics in params.get('metrics', []):
            if metrics not in eval_res:
                raise ValueError(
                    f"Metric '{metrics}' not found in evaluation results")

        return eval_res

    raise ValueError("Learner does not have an evaluate method")
