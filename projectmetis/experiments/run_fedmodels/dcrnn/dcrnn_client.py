import os

import tensorflow as tf
import numpy as np

from metisdb.metisdb_dataset_client import MetisDatasetClient
from utils.dcrnn.dcrnn_utils import StandardScaler
from utils.tf.tf_ops_dataset import TFDatasetUtils


class DcrnnDatasetClient(MetisDatasetClient):
    def __init__(self, learner_id, base_path, batch_size, pad_with_last_sample=True):
        super().__init__(learner_id)
        self._base_path = base_path
        self._batch_size = batch_size
        self._pad_with_last_sample = pad_with_last_sample
        self._scaler = self._create_scaler()

    @property
    def scaler(self):
        return self._scaler

    def parse_data_mappings_file(self, filepath, data_volume_column, csv_reader_schema,
                                 is_training=False, is_validation=False, is_testing=False):
        pass

    def generate_tfrecords(self, data_volume_column, output_filename,
                           is_training=False, is_validation=False, is_testing=False):
        if not any([is_training, is_validation, is_testing]):
            raise RuntimeError("Please indicate whether the generating tfrecord file is training/validation/testing")

        if is_training:
            filename = os.path.join(self._base_path, 'train.npz')
        elif is_validation:
            filename = os.path.join(self._base_path, 'val.npz')
        else:
            filename = os.path.join(self._base_path, 'test.npz')

        xs, ys = self._load_dataset(filename)
        num_examples = len(xs)
        tfrecords = TFDatasetUtils.serialize_to_tfrecords({'x': xs, 'y': ys}, output_filename)

        return num_examples, tfrecords

    def load_tfrecords(self, tfrecords_schema, input_filename, is_training):
        # Loads dataset from tfrecords.
        dataset = tf.data.TFRecordDataset(input_filename)

        # Parses the records into tensors.
        def _to_tensors(x):
            return TFDatasetUtils.deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecords_schema)
        dataset = dataset.map(map_func=_to_tensors, num_parallel_calls=3)

        # Bundle tensors together as key-value attributes.
        dataset = tf.data.Dataset.zip(({'x': dataset.map(map_func=lambda x, y: x),
                                        'y': dataset.map(map_func=lambda y: y)}))

        return dataset

    def x_train_input_name(self):
        return 'x'

    def y_train_output_name(self):
        return 'y'

    def x_eval_input_name(self):
        return 'x'

    def y_eval_output_name(self):
        return 'y'

    def _create_scaler(self):
        xs, ys = self._load_dataset(os.path.join(self._base_path, 'train.npz'))
        scaler = StandardScaler(mean=xs[..., 0].mean(), std=xs[..., 0].std())
        return scaler

    def _load_dataset(self, filename):
        data = np.load(filename)
        xs = np.nan_to_num(data['x'])
        ys = np.nan_to_num(data['y'])

        if self._pad_with_last_sample:
            num_padding = (self._batch_size - (len(xs) % self._batch_size)) % self._batch_size
            xs = self._pad_with_sample(xs, xs[-1:], num_padding)
            ys = self._pad_with_sample(ys, ys[-1:], num_padding)

        return xs, ys

    @staticmethod
    def _pad_with_sample(arr, sample, times):
        padding = np.repeat(sample, times, axis=0)
        arr = np.concatenate([arr, padding], axis=0)
        return arr
