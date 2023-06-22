import abc
from typing import List

import numpy as np

from metisfl.learner.weight_decrypt import ModelWeightsDescriptor


class MetisModel(object):
    @abc.abstractmethod
    def load(self, model_dir):
        pass
    
    @abc.abstractmethod
    def save(self, model_dir):
        pass
    
    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        pass
        
    @abc.abstractmethod
    def set_model_weights(self,
                          weights_names: List[str],
                          weights_trainable: List[bool],
                          weights_values: List[np.ndarray],
                          *args, **kwargs):
        pass