"""Common utilities for MetisFL."""

import random
from typing import List, Optional, Tuple, Union

import numpy as np


def iid_partition(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: Union[np.ndarray, List[np.ndarray]],
    num_partitions: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into IID chunks.

    Parameters
    ----------
    x_train : Union[np.ndarray, List[np.ndarray]]
        The training data.
    y_train : Union[np.ndarray, List[np.ndarray]]
        The training labels.
    num_partitions : int
        The number of partitions.
    seed : int, optional
        The random seed, by default 1990

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The IID chunks of the data.
    """

    idx = list(range(len(x_train)))
    random.seed(seed)
    random.shuffle(idx)

    x_train_randomized = x_train[idx]
    y_train_randomized = y_train[idx]

    chunk_size = int(len(x_train) / num_partitions)
    x_chunks, y_chunks = [], []

    for i in range(num_partitions):
        x_chunks.append(
            x_train_randomized[idx[i * chunk_size:(i + 1) * chunk_size]])
        y_chunks.append(
            y_train_randomized[idx[i * chunk_size:(i + 1) * chunk_size]])

    x_chunks = np.array(x_chunks)
    y_chunks = np.array(y_chunks)

    return x_chunks, y_chunks


def niid_partition(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: Union[np.ndarray, List[np.ndarray]],
    num_partitions: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into Non-IID chunks."""

    pass


def random_id_generator() -> int:
    """ Generates a random id. """
    # generate random id
    random_id = random.randint(0, 10000000000)

    return random_id


print(random_id_generator())
