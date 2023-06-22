
import tensorflow as tf

from metisfl.models.model_dataset import ModelDataset
from metisfl.utils.metis_logger import MetisLogger


def construct_dataset_pipeline(dataset: ModelDataset, batch_size, is_train=False):
    """
    A helper function to distinguish whether we have a tf.dataset or other data input sequence (e.g. numpy).
    We need to set up appropriately the data pipeline since keras method invocations require different parameters
    to be explicitly set. See also: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    :param dataset:
    :param batch_size:
    :param is_train:
    :return:
    """
    # We load both (x, y) variables. If variable x is not empty and is of type tf.data.Dataset,
    # then we shuffle and batch the dataset and return only a value for variable x. Otherwise,
    # we return both _x and _y assuming the two variables refer to numpy arrays or other
    # data types that are not tensorflow/keras specific.
    _x, _y = dataset.get_x(), dataset.get_y()
    
    # @stripeli this is dataset pipeline specific code. it should be moved 
    # to the dataset class, not model ops.
    if isinstance(_x, tf.data.Dataset):
        MetisLogger.info("Model dataset input is a tf.data.Dataset; ignoring fed y values.")
        if is_train:
            # Shuffle all records only if dataset is used for training.
            _x = _x.shuffle(dataset.get_size())
        # If the input is of tf.Dataset we only need to return the input x,
        # we do not need to set a value for target y.
        _x, _y = _x.batch(batch_size), None
    return _x, _y

