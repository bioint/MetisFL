import abc

import numpy as np

from projectmetis.python.models.model_dataset import ModelDataset
from projectmetis.proto import metis_pb2, model_pb2


class ModelOps(object):

    def __init__(self, model, encryption_scheme=None, *args, **kwargs):
        self._model = model
        self._encryption_scheme = encryption_scheme

    def get_model_weights_from_variables_pb(self, variables: [model_pb2.Model.Variable]):
        assert all([isinstance(var, model_pb2.Model.Variable) for var in variables])
        var_names, var_trainables, var_nps = [], [], []
        for var in variables:
            var_name = var.name
            var_trainable = var.trainable
            is_ciphertext = False
            if var.HasField('int_tensor'):
                var_tensor = var.int_tensor
            if var.HasField('double_tensor'):
                var_tensor = var.double_tensor
            # TODO Adding ciphertext_tensor support.
            if var.HasField('ciphertext_tensor'):
                var_tensor = var.ciphertext_tensor
                is_ciphertext = True

            tensor_spec = var_tensor.spec
            tensor_length = tensor_spec.length
            tensor_dims = tensor_spec.dimensions
            tensor_dtype = tensor_spec.dtype
            tensor_values = var_tensor.values
            if is_ciphertext:
                assert self._encryption_scheme is not None, "Need encryption scheme to decrypt tensor."
                tensor_values = self._encryption_scheme.decrypt(tensor_values, tensor_length, 1)

            if tensor_dtype == model_pb2.TensorSpec.DType.INT:
                    tensor_np = np.array(tensor_values, dtype=np.int)
            elif tensor_dtype == model_pb2.TensorSpec.DType.LONG:
                tensor_np = np.array(tensor_values, dtype=np.long)
            elif tensor_dtype == model_pb2.TensorSpec.DType.FLOAT:
                tensor_np = np.array(tensor_values, dtype=np.float32)
            elif tensor_dtype == model_pb2.TensorSpec.DType.DOUBLE:
                tensor_np = np.array(tensor_values, dtype=np.float64)
            else:
                # The default data type in Numpy is float64.
                tensor_np = np.array(tensor_values, dtype=np.float64)
            tensor_np = tensor_np.reshape(tensor_dims)
            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(tensor_np)
        return var_names, var_trainables, var_nps

    @abc.abstractmethod
    def load_model(self, filepath=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, filepath=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def set_model_weights(self, weights_names: [str], weights_trainable: [bool], weights_values: [np.ndarray([])],
                          *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_model_weights(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_model(self, train_dataset: ModelDataset, learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters, validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None, verbose=False,
                    *args, **kwargs) -> metis_pb2.CompletedLearningTask:
        pass

    @abc.abstractmethod
    def evaluate_model(self, dataset: ModelDataset, batch_size=100,
                       metrics=None, verbose=False, *args, **kwargs) -> metis_pb2.ModelEvaluation:
        pass

    @abc.abstractmethod
    def infer_model(self, dataset: ModelDataset, batch_size=100, *args, **kwargs):
        pass

    @abc.abstractmethod
    def construct_optimizer(self, optimizer_config_pb: model_pb2.OptimizerConfig = None, *args, **kwargs):
        pass
