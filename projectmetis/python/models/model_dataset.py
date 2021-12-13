import abc

import numpy as np


class ModelDataset(object):
    """
    Private Class. Users need to wrap their datasets as one of its subclasses.
    """

    def __init__(self, x=np.array([]), y=np.array([]), size=0):
        """
        A ModelDataset is a wrapper over the model's train/test/validation dataset input and expected output.
        However, a ModelDataset could hold just the model input since it could be wrapper over a dataset, which we
        use it just for inference.
        :param x: dataset input
        :param y: dataset ground truth
        :param size: total number of dataset records
        """
        self._x = x
        self._y = y
        self._size = size

    def get_x(self, *args, **kwargs):
        return self._x

    def get_y(self, *args, **kwargs):
        return self._y

    def get_size(self, *args, **kwargs):
        return self._size

    @abc.abstractmethod
    def get_model_dataset_specifications(self, *args, **kwargs):
        return dict()


class ModelDatasetClassification(ModelDataset):

    def __init__(self, x=np.array([]), y=np.array([]), size=0, examples_per_class=None):
        super(ModelDatasetClassification, self).__init__(x, y, size)
        self.examples_per_class = examples_per_class
        if self.examples_per_class is None:
            self.examples_per_class = dict()
        else:
            assert isinstance(self.examples_per_class, dict)

    def get_model_dataset_specifications(self, *args, **kwargs):
        return self.examples_per_class


class ModelDatasetRegression(ModelDataset):

    def __init__(self, x=np.array([]), y=np.array([]), size=0,
                 min_val=0.0, max_val=0.0,
                 mean_val=0.0, median_val=0.0,
                 mode_val=0.0, stddev_val=0.0):
        super(ModelDatasetRegression, self).__init__(x, y, size)
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

