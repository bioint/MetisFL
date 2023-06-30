import abc

from metisfl.models.types import ModelWeightsDescriptor

class MetisModel(object):
    nn_engine = None
        
    @abc.abstractmethod
    def load(self, model_dir) -> "MetisModel":
        pass
    
    @abc.abstractmethod
    def save(self, model_dir) -> None:
        pass
    
    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        pass
        
    @abc.abstractmethod
    def set_model_weights(self, model_weights_descriptor: ModelWeightsDescriptor):
        pass
    
    @abc.abstractmethod
    def get_neural_engine(self):
        assert self.nn_engine is not None, "Neural engine not set"
        return self.nn_engine 