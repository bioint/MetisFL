"""Common utilities for MetisFL."""

import random
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


def iid_partition(
    x_train: Iterable,
    num_partitions: int,
    y_train: Iterable = None,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into IID chunks.

    Parameters
    ----------
    x_train : Iterable
        An iterable containing the data to be partitioned.
    num_partitions : int
        The number of partitions.
    y_train : Iterable, optional
        The labels, by default None
    seed : int, optional
        The random seed, by default 1990

    Returns
    -------
    Tuple[Iterable, Iterable]
        The IID chunks of the data.
    """

    random.seed(seed)

    chunk_size = int(len(x_train) / num_partitions)
    x_chunks = []
    if y_train is not None:
        y_chunks = []

    for i in range(num_partitions):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        x_chunks.append(x_train[start:end])

        if y_train is not None:
            y_chunks.append(y_train[start:end])

    x_chunks = np.array(x_chunks)
    if y_train is not None:
        y_chunks = np.array(y_chunks)

    if y_train is not None:
        return x_chunks, y_chunks

    return x_chunks


def niid_partition(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: Union[np.ndarray, List[np.ndarray]],
    num_partitions: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into Non-IID chunks."""

    pass
