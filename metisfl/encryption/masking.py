from metisfl.encryption.encryption import Encryption
from metisfl.proto import model_pb2
from metisfl.models.types import ModelWeightsDescriptor

class Masking(Encryption):
    
    def decrypt_model(self, model_pb: model_pb2.Model) -> ModelWeightsDescriptor:
        pass

    def encrypt_model(self, weights_descriptor: ModelWeightsDescriptor) -> model_pb2.Model:
        pass