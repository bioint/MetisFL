import abc
import torch

import numpy as np
import tensorflow as tf

from collections import namedtuple
from metisfl.models.model_dataset import ModelDataset
from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.proto_messages_factory import ModelProtoMessages
from typing import List

ModelWeightsDescriptor = \
    namedtuple('ModelWeightsDescriptor',
               ['nn_engine', 'weights_names', 'weights_trainable', 'weights_values'])


class ModelOps(object):

    def __init__(self, model, he_scheme=None, *args, **kwargs):
        self._model = model
        self._he_scheme = he_scheme

    def get_model_weights_from_variables_pb(self, variables: [model_pb2.Model.Variable]):
        assert all([isinstance(var, model_pb2.Model.Variable) for var in variables])
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in variables:
            # Variable specifications.
            var_name = var.name
            var_trainable = var.trainable

            if var.HasField("ciphertext_tensor"):
                assert self._he_scheme is not None, "Need encryption scheme to decrypt tensor."
                # For a ciphertext tensor, first we need to decrypt it, and then load it
                # into a numpy array with the data type specified in the tensor specifications.
                tensor_spec = var.ciphertext_tensor.tensor_spec
                tensor_length = tensor_spec.length
                decoded_value = self._he_scheme.decrypt(tensor_spec.value, tensor_length, 1)
                # Since the tensor is decoded we just need to recreate the numpy array
                # to its original data type and shape.
                np_array = \
                    ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                        tensor_spec, decoded_value)
            elif var.HasField('plaintext_tensor'):
                tensor_spec = var.plaintext_tensor.tensor_spec
                # If the tensor is a plaintext tensor, then we need to read the byte buffer
                # and load the tensor as a numpy array casting it to the specified data type.
                np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(tensor_spec)
            else:
                raise RuntimeError("Not a supported tensor type.")

            # Append variable specifications to model's variable list.
            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        return var_names, var_trainables, var_nps

    @abc.abstractmethod
    def cleanup(self):
        pass

    @abc.abstractmethod
    def load_model(self, model_dir=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, model_dir=None, *args, **kwargs):
        pass

    @abc.abstractmethod
    def set_model_weights(self,
                          weights_names: List[str],
                          weights_trainable: List[bool],
                          weights_values: List[np.ndarray],
                          *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_model_weights(self, *args, **kwargs) -> [str, List[str], List[bool], List[np.ndarray]]:
        """
            This function will return all the weights of the current model state as a tuple of lists,
            where the first list will store to the weights as numpy arrays, the second list will store
            whether each weight is trainable or not and the final list will store the name of the weight.
            :param args:
            :param kwargs:
            :return:
        """
        nn_engine = ""
        weights_names, weights_trainable, weights_values = [], [], []
        if isinstance(self._model, tf.keras.Model):
            nn_engine = "keras"
            weights_names = [w.name for layer in self._model.layers for w in layer.weights]
            all_trainable_weights_names = [v.name for v in self._model.trainable_variables]
            weights_trainable = [True if w_n in all_trainable_weights_names else False for w_n in weights_names]
            weights_values = [w.numpy() for w in self._model.weights]
        elif isinstance(self._model, torch.nn.Module):
            nn_engine = "pytorch"
            for name, param in self._model.named_parameters():
                weights_names.append(name)
                # Trainable variables require gradient computation,
                # for non-trainable the gradient is False.
                weights_trainable.append(param.requires_grad)
                weights_values.append(param.data.numpy(force=True))
        else:
            raise RuntimeError("Not a supported model type.")

        return ModelWeightsDescriptor(nn_engine=nn_engine,
                                      weights_names=weights_names,
                                      weights_trainable=weights_trainable,
                                      weights_values=weights_values)

    @abc.abstractmethod
    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False,
                    *args, **kwargs) -> metis_pb2.CompletedLearningTask:
        pass

    @abc.abstractmethod
    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       metrics=None,
                       verbose=False,
                       *args, **kwargs) -> metis_pb2.ModelEvaluation:
        pass

    @abc.abstractmethod
    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100,
                    *args, **kwargs):
        pass

    @abc.abstractmethod
    def construct_optimizer(self,
                            optimizer_config_pb: model_pb2.OptimizerConfig = None,
                            *args, **kwargs):
        pass
