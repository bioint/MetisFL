import os
from typing import List

from metisfl import config
from metisfl.encryption import fhe
from metisfl.models.types import ModelWeightsDescriptor
from metisfl.proto import metis_pb2, model_pb2
from metisfl.proto.proto_messages_factory import ModelProtoMessages


class HomomorphicEncryption(object):

    def __init__(self, batch_size: int, scaling_factor_bits: int):
        self._he_scheme = None
        self._he_scheme = fhe.CKKS(
            batch_size,
            scaling_factor_bits)
        self._setup_fhe()

    def _setup_fhe(self):
        paths = config.get_fhe_resources()
        # check if the resources are available
        bools = map(lambda path: os.path.exists(path), paths)
        if any(bools):
            self._he_scheme.load_context_and_keys_from_files(*paths[:3])
        else:
            fhe_dir = config.get_fhe_dir()
            self._he_scheme.gen_crypto_context_and_keys(fhe_dir)

    def decrypt(self,
                variables: List[model_pb2.Model.Variable]) -> ModelWeightsDescriptor:
        assert all([isinstance(var, model_pb2.Model.Variable)
                   for var in variables])
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in variables:
            var_name = var.name
            var_trainable = var.trainable

            if var.HasField("ciphertext_tensor"):
                assert self._he_scheme is not None, "Need encryption scheme to decrypt tensor."
                tensor_spec = var.ciphertext_tensor.tensor_spec
                decoded_value = self._he_scheme.decrypt(
                    tensor_spec.value, tensor_spec.length, 1)
                np_array = \
                    ModelProtoMessages.TensorSpecProto.proto_tensor_spec_with_list_values_to_numpy_array(
                        tensor_spec, decoded_value)
            elif var.HasField('plaintext_tensor'):
                tensor_spec = var.plaintext_tensor.tensor_spec
                np_array = ModelProtoMessages.TensorSpecProto.proto_tensor_spec_to_numpy_array(
                    tensor_spec)
            else:
                raise RuntimeError("Not a supported tensor type.")

            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        return ModelWeightsDescriptor(weights_names=var_names,
                                      weights_trainable=var_trainables,
                                      weights_values=var_nps)

    def encrypt(self, weights_descriptor: ModelWeightsDescriptor) -> List[model_pb2.Model]:
        weights_names = weights_descriptor.weights_names
        weights_trainable = weights_descriptor.weights_trainable
        weights_values = weights_descriptor.weights_values
        if not weights_names:
            # Populating weights names with surrogate keys.
            weights_names = ["arr_{}".format(widx)
                             for widx in range(len(weights_values))]
        if weights_trainable:
            # Since weights have not specified as trainable or not, we default all weights to trainable.
            weights_trainable = [True for _ in range(len(weights_values))]

        variables_pb = []
        for w_n, w_t, w_v in zip(weights_names, weights_trainable, weights_values):
            ciphertext = None
            if self._he_scheme:
                ciphertext = self._he_scheme.encrypt(w_v.flatten())
            # If we have a ciphertext we prioritize it over the plaintext.
            # @stripeli what does this mean?
            tensor_pb = ModelProtoMessages.construct_tensor_pb(nparray=w_v,
                                                               ciphertext=ciphertext)
            model_var = ModelProtoMessages.construct_model_variable_pb(name=w_n,
                                                                       trainable=w_t,
                                                                       tensor_pb=tensor_pb)
            variables_pb.append(model_var)

        return model_pb2.Model(
            variables=variables_pb)
