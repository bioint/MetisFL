import abc

from typing import List

from metisfl.models.model_dataset import ModelDataset
from metisfl.models.metis_model import MetisModel
from metisfl.models.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.proto import metis_pb2


class ModelOps(object):
        
    def get_model(self) -> MetisModel:
        assert self._metis_model is not None, "Model is not initialized."
        return self._metis_model

    @abc.abstractmethod
    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False) -> List[ModelWeightsDescriptor, LearningTaskStats]:
        pass
    
    @abc.abstractmethod
    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       metrics=None,
                       verbose=False) -> dict:
        pass
    
    @abc.abstractmethod
    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100):
        pass
