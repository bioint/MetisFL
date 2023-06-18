import abc
import torch
import tensorflow as tf

from typing import Dict


class ModelDef:

    @abc.abstractmethod
    def get_model(self, *args, **kwargs):
        pass


class PyTorchDef(ModelDef):

    @abc.abstractmethod
    def fit(self, dataset: torch.utils.data.DataLoader, epochs: int, *args, **kwargs) -> Dict:
        pass

    @abc.abstractmethod
    def evaluate(self, dataset: torch.utils.data.DataLoader, *args, **kwargs) -> Dict:
        pass
