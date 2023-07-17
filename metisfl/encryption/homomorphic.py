import os

from metisfl import config

from metisfl.encryption.encryption import Encryption
from metisfl.utils.metis_logger import MetisLogger
from metisfl.encryption import fhe
from metisfl.models.types import ModelWeightsDescriptor
from metisfl.proto import metis_pb2, model_pb2
from metisfl.proto.proto_messages_factory import ModelProtoMessages


class Homomorphic(Encryption):

    def __init__(self, he_scheme_pb: metis_pb2.HESchemeConfig):
        assert isinstance(
            he_scheme_pb, metis_pb2.HESchemeConfig), "Need a valid HE scheme protobuf."

        self._he_scheme = None
        if he_scheme_pb and he_scheme_pb.HasField("ckks_scheme_config"):
            self._he_scheme = fhe.CKKS(
                he_scheme_pb.ckks_scheme_config.batch_size,
                he_scheme_pb.ckks_scheme_config.scaling_factor_bits)
        else:
            MetisLogger.fatal("Not a supported homomorphic encryption scheme: {}".format(he_scheme_pb))
        
    def decrypt_model(self, model_pb: model_pb2.Model) -> ModelWeightsDescriptor:
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in model_pb.variables:
            var_name = var.name
            var_trainable = var.trainable
            tensor_spec = var.ciphertext_tensor.tensor_spec
            decoded_value = self._he_scheme.decrypt(
                tensor_spec.value, tensor_spec.length)
            np_array = \
                ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                    tensor_spec, decoded_value)
            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        return ModelWeightsDescriptor(weights_names=var_names,
                                      weights_trainable=var_trainables,
                                      weights_values=var_nps)

    def encrypt_model(self, weights_descriptor: ModelWeightsDescriptor) -> model_pb2.Model:
        weights_names = weights_descriptor.weights_names
        weights_trainable = weights_descriptor.weights_trainable
        weights_values = weights_descriptor.weights_values
        variables_pb = []
        for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):            
            ciphertext = self._he_scheme.encrypt(w_v.flatten())            
            tensor_pb = ModelProtoMessages.construct_tensor_pb(nparray=w_v,
                                                               ciphertext=ciphertext)
            model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                       trainable=w_t,
                                                                       tensor_pb=tensor_pb)
            variables_pb.append(model_var)

        return model_pb2.Model(
            variables=variables_pb)

    def initialize_crypto_params(self, crypto_dir):
        self._he_scheme.gen_crypto_context_and_keys(crypto_dir)

    def _decrypt_data(self, ciphertext: str, num_elems: int):
        return self._he_scheme.decrypt(ciphertext, num_elems)

    def _encrypt_data(self, values):      
        return self._he_scheme.encrypt(values)
