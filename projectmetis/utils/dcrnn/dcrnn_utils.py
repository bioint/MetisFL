import pickle

import numpy as np
import tensorflow as tf
from absl import logging


# ============== #
# Loss Functions #
# ============== #

def __get_mask(labels, null_val):
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    return mask


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    with tf.variable_scope('masked_mae'):
        mask = __get_mask(labels, null_val)
        loss = tf.abs(tf.subtract(preds, labels))
        loss = loss * mask
        loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
        return tf.reduce_mean(loss)


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss


# ============== #
# Data Utilities #
# ============== #

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ============= #
# I/O Utilities #
# ============= #

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        logging.exception('Unable to load pickle data: %s', pickle_file)
        raise e
