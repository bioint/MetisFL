import abc
from collections import namedtuple
from typing import List


from metisfl.models.model_dataset import ModelDataset
from metisfl.models.model_wrapper import MetisModel, ModelWeightsDescriptor
from metisfl.proto import metis_pb2

LearningTaskStats = namedtuple('LearningTaskStats', [
    "train_stats",
    "completed_epochs",
    "global_iteration",
    "validation_stats",
    "test_stats",
    "completes_batches",
    "batch_size",
    "processing_ms_per_epoch",
    "processing_ms_per_batch"
])

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
                    verbose=False) -> [ModelWeightsDescriptor, LearningTaskStats]
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
