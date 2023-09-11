import tensorflow as tf
from typing import Tuple

from metisfl.common.utils import iid_partition


def load_data(max_learners: int = 3, rescale_reshape: bool = True) -> Tuple:
    """A helper function to load the Fashion MNIST dataset.

    Args:
        rescale_reshape (bool, optional): Whether to rescale and reshape. (Defaults to True)

    Returns:
        Tuple: A tuple containing the training and test data.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize and reshape the data
    if rescale_reshape:
        x_train = (x_train.astype('float32') / 256).reshape(-1, 32, 32, 3)
        x_test = (x_test.astype('float32') / 256).reshape(-1, 32, 32, 3)

     # Partition the data into 3 clients, iid
    x_chunks, y_chunks = iid_partition(
        x_train=x_train, y_train=y_train, num_partitions=max_learners)

    return x_chunks, y_chunks, x_test, y_test
