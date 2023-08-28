import os

import numpy as np
import tensorflow as tf

from typing import List, Tuple

from metisfl.common.data_partitioning import DataPartitioning


def load_data(rescale_reshape=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A helper function to load the Fashion MNIST dataset.

    Args:
        rescale_reshape (bool, optional): Whether to rescale and reshape. (Defaults to True)

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray): A tuple containing the training and test data.
    """

    # Load data using TensorFlow Keras API
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize and reshape the data
    if rescale_reshape:
        x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
        x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

    # Return the data
    return x_train, y_train, x_test, y_test


def partition_data_iid(x_train, y_train, num_learners):
    x_chunks, y_chunks = DataPartitioning(
        x_train, y_train, num_learners).iid_partition()
    return x_chunks, y_chunks


def partition_data_noniid(x_train, y_train, num_learners):
    x_chunks, y_chunks = DataPartitioning(
        x_train, y_train, num_learners).non_iid_partition()
    return x_chunks, y_chunks


def save_data(x_chunks, y_chunks, x_test, y_test) -> Tuple[List[str], str]:
    """Saves the data to disk using the `np.savez` function.

    Args:
        x_chunks (np.ndarray): The x values for each learner.
        y_chunks (np.ndarray): The y values for each learner.
        x_test (np.ndarray): The x values for the test dataset.
        y_test (np.ndarray): The y values for the test dataset.

    Returns:
        tuple(list(str), str): A tuple containing the file paths for the training and test datasets.
    """
    # Get the directory of this script
    script_cwd = os.path.dirname(__file__)

    # Create a `data` directory relative to this script
    datasets_path = os.path.join(script_cwd, "data/")
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # A list to store the training data file paths for each learner
    train_dataset_fps = []

    # Save training data for each learner
    for cidx, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
        filepath = os.path.join(datasets_path, "train_{}.npz".format(cidx))
        np.savez(filepath, x=x_chunk, y=y_chunk)
        train_dataset_fps.append(filepath)

    # Save test data
    test_dataset_fp = os.path.join(datasets_path, "test.npz")
    np.savez(test_dataset_fp, x=x_test, y=y_test)

    # Return the file paths
    return train_dataset_fps, test_dataset_fp
