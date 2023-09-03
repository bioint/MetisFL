"""Common utilities for MetisFL."""

import random
from typing import List, Optional, Tuple, Union

import numpy as np


def iid_partition(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: Union[np.ndarray, List[np.ndarray]],
    partitions_num: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into IID chunks.

    Parameters
    ----------
    x_train : Union[np.ndarray, List[np.ndarray]]
        The training data.
    y_train : Union[np.ndarray, List[np.ndarray]]
        The training labels.
    partitions_num : int
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

    chunk_size = int(len(x_train) / partitions_num)
    x_chunks, y_chunks = [], []

    for i in range(partitions_num):
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
    partitions_num: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into Non-IID chunks."""

    pass
