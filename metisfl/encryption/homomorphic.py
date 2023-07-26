import os

from typing import Any, Dict, List

from metisfl.encryption.encryption_scheme import EncryptionScheme
from metisfl.proto.proto_messages_factory import MetisProtoMessages
from metisfl.utils.logger import MetisLogger
from metisfl.encryption import fhe
from metisfl.proto import metis_pb2


class Homomorphic(EncryptionScheme):

    def __init__(self, he_scheme_pb: metis_pb2.HEScheme, init_crypto_params=False):
        assert isinstance(he_scheme_pb, metis_pb2.HEScheme), \
            "Need a valid HE scheme protobuf."
        super().__init__(init_crypto_params)

        self._he_scheme_config_pb = he_scheme_pb.config
        self._he_scheme_pb = he_scheme_pb
        self._he_scheme = None
        if he_scheme_pb and he_scheme_pb.HasField("ckks_scheme"):
            self._he_scheme = fhe.CKKS(
                he_scheme_pb.ckks_scheme.batch_size,
                he_scheme_pb.ckks_scheme.scaling_factor_bits)            
        else:
            MetisLogger.fatal(
                "Not a supported homomorphic encryption scheme: {}".format(
                he_scheme_pb))
        
        if init_crypto_params:
            # If we need to initialize the crypto params, we need to
            # override 
            self._he_scheme_config_pb = self._generate_crypto_params()
        self._load_crypto_params()

    def decrypt_data(self, ciphertext: str, num_elems: int) -> List[Any]:
        return self._he_scheme.decrypt(ciphertext, num_elems)

    def encrypt_data(self, values) -> Any:
        return self._he_scheme.encrypt(values)

    def to_proto(self) -> metis_pb2.EncryptionScheme:
        he_scheme_pb = None
        if self._he_scheme_pb.HasField("ckks_scheme"):
            he_scheme_pb = MetisProtoMessages.construct_he_scheme_pb(
                config_pb=self._he_scheme_config_pb,
                ckks_scheme_pb=self._he_scheme_pb.ckks_scheme)        
        return MetisProtoMessages.construct_encryption_scheme_pb(
            he_scheme_pb=he_scheme_pb)

    def _generate_crypto_params(self) -> metis_pb2.HESchemeConfig:
        cc, pubk, pk = self._he_scheme.gen_crypto_params(
            self._he_scheme_pb.config.crypto_context,
            self._he_scheme_pb.config.public_key,
            self._he_scheme_pb.config.private_key)
        return MetisProtoMessages.construct_he_scheme_config(cc, pubk, pk)

    def _load_crypto_params(self):
        # We only load crypto params when they are defined.
        if self._he_scheme_config_pb.crypto_context:
            self._he_scheme.load_crypto_context(
                self._he_scheme_config_pb.crypto_context)
        if self._he_scheme_config_pb.public_key:
            self._he_scheme.load_public_key(
                self._he_scheme_config_pb.public_key)
        if self._he_scheme_config_pb.private_key:
            self._he_scheme.load_private_key(
                self._he_scheme_config_pb.private_key)
