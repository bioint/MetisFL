import abc

from metisfl.models.types import ModelWeightsDescriptor


class MetisModel(object):
    _nn_engine = None
    _backend_model = None

    @staticmethod
    @abc.abstractmethod
    def load(self, model_dir) -> "MetisModel":
        pass

    @abc.abstractmethod
    def save(self, model_dir, initialize=False) -> None:
        pass

    @abc.abstractmethod
    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        pass

    @abc.abstractmethod
    def set_model_weights(self, model_weights_descriptor: ModelWeightsDescriptor):
        pass

    def get_neural_engine(self):
        assert self._nn_engine is not None, "Neural engine not set."
        return self._nn_engine
