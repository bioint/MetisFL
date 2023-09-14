"""Common utilities for MetisFL."""

import random, os, shutil
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import glob

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


def iid_partition_dir(
    path: str,
    extension: str,
    num_partitions: int,
    subdir_name: str = None
) -> None:
    """ Partitions all files in a dir into IID chunks and saves them in subdirs named chunk0, chunk1, etc.

    Parameters
    ----------
    path : str
        The path to the directory containing the files to be partitioned.
    extension : str
        The extension of the files to be partitioned.
    num_partitions : int
        The number of partitions.
    subdir_name : str, optional
        If given, the files will be placed under `chunk0/subdir_name`, `chunk1/subdir_name`, etc.
    """
    files = glob.glob(path + "/*." + extension)
    num_files = len(files)
    chunk_size = int(num_files / num_partitions)
    random.shuffle(files)

    paths = []
    for i in range(num_partitions):
        if not subdir_name:
            paths.append(path + "/chunk" + str(i))
        else:
            paths.append(path + "/chunk" + str(i) + "/" + subdir_name)

    for i in range(num_partitions):
        os.makedirs(paths[i], exist_ok=True)

    for i in range(num_partitions):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        for j in range(start, end):
            shutil.move(files[j], paths[i])
            

def iid_repartition_dir(
    path: str,
    num_partitions: int,
    chunk_prefix: str = "chunk",
) -> None:
    """ Repartition all files in the chuck-ed dir into IID chunks and saves them into new subdirs named chunk0, chunk1, etc.
        This is useful when you want to change the number of partitions.
        
    Parameters
    ----------
    path : str
        The path to the directory containing the files to be partitioned.
    num_partitions : int
        The number of partitions.
    chunk_prefix : str
        The prefix of the subdirs containing the files to be repartitioned.     
    """    
    pass

def niid_partition(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: Union[np.ndarray, List[np.ndarray]],
    num_partitions: int,
    seed: Optional[int] = 1990,
) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions the data into Non-IID chunks."""

    pass
