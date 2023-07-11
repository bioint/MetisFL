import abc

from metisfl.models.types import ModelWeightsDescriptor
from metisfl.models.utils import construct_model_pb, get_weights_from_model_pb


class MetisModel(object):
    _nn_engine = None
    _backend_model = None

    @abc.abstractmethod
    @staticmethod
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

    def get_metis_model_pb(self, 
                           model_weights_descriptor: ModelWeightsDescriptor,
                           he_scheme_pb: metis_pb2.HESchemeConfig) -> metis_pb2.Model:
        construct_model_pb(model_weights_descriptor, he_scheme_pb)

    def get_metis_model_weights(self, 
                                model_pb: metis_pb2.Model,
                                he_scheme_pb: metis_pb2.HESchemeConfig) -> ModelWeightsDescriptor:
        get_weights_from_model_pb(model_pb, he_scheme_pb)
