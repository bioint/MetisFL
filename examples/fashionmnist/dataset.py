import os

import numpy as np
import tensorflow as tf

from metisfl.utils.data_partitioning import DataPartitioning


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
    x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

def partition_data_iid(x_train, y_train, num_learners):
    x_chunks, y_chunks = DataPartitioning(x_train, y_train, num_learners).iid_partition()
    return x_chunks, y_chunks

def partition_data_noniid(x_train, y_train, num_learners):
    x_chunks, y_chunks = DataPartitioning(x_train, y_train, num_learners).noniid_partition()
    return x_chunks, y_chunks

def save_data():
    script_cwd = os.path.dirname(__file__)
    datasets_path = os.path.join(script_cwd, "datasets/fashionmnist/")
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)
    np.savez(os.path.join(datasets_path, "test.npz"), x=x_test, y=y_test)
    for cidx, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
        np.savez(os.path.join(datasets_path, "train_{}.npz".format(cidx)), x=x_chunk, y=y_chunk)
    for lidx, learner in enumerate(federation_environment.learners.learners):
        learner.dataset_configs.test_dataset_path = \
            os.path.join(datasets_path, "test.npz")
        learner.dataset_configs.train_dataset_path = \
            os.path.join(datasets_path, "train_{}.npz".format(lidx))
