

from typing import Dict, List


def dataset_scaling(num_training_examples: Dict[str, int]) -> Dict[str, int]:
    """Computes the scaling factor for the given learners based on the number of training examples.

    Parameters
    ----------
    num_training_examples : Dict[str, int]
        The number of training examples for each learner.

    Returns
    -------
    Dict[str, float]
        The scaling factor for each learner.
    """

    total_examples = sum(num_training_examples.values())
    scaling_factor = {}

    for learner_id, num_examples in num_training_examples.items():
        scaling_factor[learner_id] = num_examples / total_examples

    return scaling_factor


def participants_scaling(learner_ids: List[str]) -> Dict[str, float]:
    """Computes the scaling factor for the given learners based on the number of learners.

    Parameters
    ----------
    learner_ids : List[str]
        The learner ids.

    Returns
    -------
    Dict[str, float]
        The scaling factor for each learner.
    """

    scaling_factor = {}

    for learner_id in learner_ids:
        scaling_factor[learner_id] = 1 / len(learner_ids)

    return scaling_factor


def batches_scaling(num_completed_batches: Dict[str, int]) -> Dict[str, int]:
    """Computes the scaling factor for the given learners based on the number of completed batches.

    Parameters
    ----------
    num_completed_batches : Dict[str, int]
        The number of completed batches for each learner.

    Returns
    -------
    Dict[str, float]
        The scaling factor for each learner.
    """

    total_batches = sum(num_completed_batches.values())
    scaling_factor = {}

    for learner_id, num_batches in num_completed_batches.items():
        scaling_factor[learner_id] = num_batches / total_batches

    return scaling_factor
