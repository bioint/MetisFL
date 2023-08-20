
"""MetisFL Homomorphic Encryption Module using Palisade."""

import os
from typing import List

from metisfl import config
from metisfl.encryption import fhe
from ..models.types import ModelWeightsDescriptor
from ..proto import model_pb2
from ..proto.proto_messages_factory import ModelProtoMessages


class HomomorphicEncryption(object):

    """Homomorphic Encryption class using Palisade. Wraps the C++ implementation of Palisade."""

    def __init__(
        self,
        batch_size: int,
        scaling_factor_bits: int,
    ):
        """Initializes the HomomorphicEncryption object. 

        Parameters
        ----------
        batch_size : int
            The batch size of the encryption scheme.
        scaling_factor_bits : int
            The number of bits to use for the scaling factor.
        crypto_context_path : str, optional
            The path to the crypto context file, by default None

        """
        self._he_scheme = fhe.CKKS(batch_size, scaling_factor_bits)
        self._setup_fhe()

    def _setup_fhe(self):
        """Sets up the FHE scheme."""

        paths = config.get_fhe_resources()
        bools = map(lambda path: os.path.exists(path), paths)

        if any(bools):
            self._he_scheme.load_context_and_keys_from_files(*paths[:3])
        else:
            fhe_dir = config.get_fhe_dir()
            self._he_scheme.gen_crypto_context_and_keys(fhe_dir)

    def decrypt(self, model: model_pb2.Model) -> model_pb2.Model:

        for var in model.variables:
            if var.encryped:
                decoded_value = self._he_scheme.decrypt(
                    var.tensor.value, var.tensor.length, 1)
                var.tensor.value = decoded_value
                var.encrypted = False

        variables = model.variables
        var_names, var_trainables, var_nps = list(), list(), list()
        for var in variables:
            var_name = var.name
            var_trainable = var.trainable

            if var.HasField("ciphertext_tensor"):
                assert self._he_scheme is not None, "Need encryption scheme to decrypt tensor."
                tensor = var.ciphertext_tensor.tensor
                decoded_value = self._he_scheme.decrypt(
                    tensor.value, tensor.length, 1)
                np_array = \
                    ModelProtoMessages.tensorsProto.proto_tensor_spec_with_list_values_to_numpy_array(
                        tensor, decoded_value)
            elif var.HasField('plaintext_tensor'):
                tensor = var.plaintext_tensor.tensor
                np_array = ModelProtoMessages.tensorsProto.proto_tensor_spec_to_numpy_array(
                    tensor)
            else:
                raise RuntimeError("Not a supported tensor type.")

            var_names.append(var_name)
            var_trainables.append(var_trainable)
            var_nps.append(np_array)

        return ModelWeightsDescriptor(weights_names=var_names,
                                      weights_trainable=var_trainables,
                                      weights_values=var_nps)

    def encrypt(self, model: model_pb2.Model) -> model_pb2.Model:
        # FIXME:
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
