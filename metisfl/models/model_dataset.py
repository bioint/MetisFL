import tensorflow as tf

from metisfl.utils.metis_logger import MetisLogger


class ModelDataset(object):
    """
    Private Class. Users need to wrap their datasets as one of its subclasses.
    """

    def __init__(self, x, y=None):
        """
        A ModelDataset is a wrapper over the model's train/test/validation dataset input and expected output.
        :param dataset: dataset input
        :param x: model input
        :param y: output
        """
        assert x is not None, "ModelDataset: x cannot be None"
        self._x = x
        if isinstance(x, tf.data.Dataset):
            MetisLogger.info("Model dataset input is a tf.data.Dataset; ignoring fed y values.")
        else: 
            self._y = y
        self._size = len(x)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_size(self):
        return self._size

    def construct_dataset_pipeline(self, batch_size, is_train=False):
        _x, _y = self._x, self._y
        if isinstance(self.x, tf.data.Dataset):
            if is_train:
                # Shuffle all records only if dataset is used for training.
                _x = _x.shuffle(self.get_size())
            # If the input is of tf.Dataset we only need to return the input x,
            # we do not need to set a value for target y.
            _x, _y = _x.batch(batch_size), None
        return _x, _y


class ModelDatasetClassification(ModelDataset):

    def __init__(self, x=None, y=None, size=0, examples_per_class=None):
        super(ModelDatasetClassification, self).__init__(x=x, y=y, size=size)
        self.examples_per_class = examples_per_class
        if self.examples_per_class is None:
            self.examples_per_class = dict()
        else:
            assert isinstance(self.examples_per_class, dict)

    def get_model_dataset_specifications(self, *args, **kwargs):
        return self.examples_per_class


class ModelDatasetRegression(ModelDataset):

    def __init__(self, x=None, y=None,
                 size=0, min_val=0.0, max_val=0.0,
                 mean_val=0.0, median_val=0.0,
                 mode_val=0.0, stddev_val=0.0):
        super(ModelDatasetRegression, self).__init__(x=x, y=y, size=size)
        self.min_val = min_val
        self.max_val = max_val
        self.mean_val = mean_val
        self.median_val = median_val
        self.mode_val = mode_val
        self.stddev = stddev_val

    def get_model_dataset_specifications(self, *args, **kwargs):
        regression_specs = dict()
        regression_specs["min"] = self.min_val
        regression_specs["max"] = self.max_val
        regression_specs["mean"] = self.mean_val
        regression_specs["median"] = self.median_val
        regression_specs["mode"] = self.mode_val
        regression_specs["stddev"] = self.stddev
        return regression_specs
