import torch

from projectmetis.proto import metis_pb2
from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.models.model_dataset import ModelDataset
from projectmetis.python.models.model_ops import ModelOps


class PyTorchModelOps(ModelOps):

    def __init__(self, model_filepath="/tmp/model", *args, **kwargs):
        self._model_filepath = model_filepath
        self._model = self.load_model(self._model_filepath)
        super(PyTorchModelOps, self).__init__(self._model)

    def load_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Loading model from: {}".format(filepath))
        m = torch.load(filepath)
        MetisLogger.info("Loaded model from: {}".format(filepath))
        return m

    def save_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Saving model to: {}".format(filepath))
        torch.save(self._model, filepath=filepath)
        MetisLogger.info("Saved model at: {}".format(filepath))

    def set_model_weights(self, weights=None, *args, **kwargs):
        pass

    def get_model_weights(self, *args, **kwargs):
        pass

    def train_model(self, train_dataset: ModelDataset, total_steps=100,
                    batch_size=100, validation_dataset: ModelDataset = None,
                    verbose=False, *args, **kwargs):
        pass

    def evaluate_model(self, dataset: ModelDataset = None, batch_size=100,
                       metrics=None, verbose=False, *args, **kwargs):
        pass

    def infer_model(self, dataset: ModelDataset = None, batch_size=100, *args, **kwargs):
        pass

    def construct_optimizer(self, hyperparameters_pb: metis_pb2.Hyperparameters = None, *args, **kwargs):
        pass
