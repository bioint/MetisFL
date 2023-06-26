import abc
from collections import namedtuple
from typing import List


from metisfl.models.model_dataset import ModelDataset
from metisfl.models.model_wrapper import MetisModel
from metisfl.proto import metis_pb2, model_pb2


class ModelOps(object):
    def get_model(self) -> MetisModel:
        return self._model

    @abc.abstractmethod
    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False) -> metis_pb2.CompletedLearningTask:
        pass
    
    @abc.abstractmethod
    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       metrics=None,
                       verbose=False) -> metis_pb2.ModelEvaluation:
        pass
    
    @abc.abstractmethod
    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100):
        pass
