import abc

from typing import Dict

from metisfl.encryption.homomorphic import Homomorphic
from metisfl.encryption.masking import Masking
from metisfl.models.types import ModelWeightsDescriptor
from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.metis_logger import MetisLogger


class Encryption(object):

    @staticmethod
    def from_proto(encryption_scheme: metis_pb2.EncryptionScheme):
        if encryption_scheme.HasField("he_scheme"):
            return Homomorphic(encryption_scheme.he_scheme)
        elif encryption_scheme.HasField("masking_scheme"):
            return Masking(encryption_scheme.masking_scheme)
        else:
            MetisLogger.fatal(
                "Not a supported encryption scheme: {}".format(
                encryption_scheme))
    

    @abc.abstractmethod
    def decrypt_model(self, model_pb: model_pb2.Model) -> ModelWeightsDescriptor:    
        pass

    @abc.abstractmethod
    def encrypt_model(self, weights_descriptor: ModelWeightsDescriptor) -> model_pb2.Model:
        pass

    @abc.abstractmethod
    def initialize_crypto_params(self, crypto_dir) -> Dict:
        pass
