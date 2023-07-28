from typing import Any, List

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

        self._he_scheme_config_pb = he_scheme_pb.he_scheme_config
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
            # If we need to initialize the crypto params, we need 
            # to override the previous he scheme configuration with 
            # the newly generated crypto param values.
            self._he_scheme_config_pb = self._generate_crypto_params()
        
        # Irrespective of whether new crypto params were generated or
        # whether we are using the existing crypto params, we still 
        # need to load the parameters into the crypto scheme.
        self._load_crypto_params()

    def decrypt_data(self, ciphertext: str, num_elems: int) -> List[Any]:
        return self._he_scheme.decrypt(ciphertext, num_elems)

    def encrypt_data(self, values) -> Any:
        return self._he_scheme.encrypt(values)

    def to_proto(self) -> metis_pb2.EncryptionConfig:
        he_scheme_pb = None
        if self._he_scheme_pb.HasField("ckks_scheme"):
            he_scheme_pb = MetisProtoMessages.construct_he_scheme_pb(
                he_scheme_config_pb=self._he_scheme_config_pb,
                scheme_pb=self._he_scheme_pb.ckks_scheme)        
        return MetisProtoMessages.construct_encryption_config_pb(
            he_scheme_pb=he_scheme_pb)

    def _generate_crypto_params(self) -> metis_pb2.HESchemeConfig:
        if self._he_scheme_config_pb.as_files:
            self._he_scheme.gen_crypto_params_files(
                self._he_scheme_pb.he_scheme_config.crypto_context, 
                self._he_scheme_pb.he_scheme_config.public_key, 
                self._he_scheme_pb.he_scheme_config.private_key)
            crypto_params = self._he_scheme.get_crypto_params_files()
        else:
            crypto_params = self._he_scheme.gen_crypto_params()
            # crypto_params = { param: val \
            #                  for param, val in crypto_params.items() }
        
        return MetisProtoMessages.construct_he_scheme_config_pb(
                    self._he_scheme_config_pb.as_files,
                    crypto_params["crypto_context"],
                    crypto_params["public_key"],
                    crypto_params["private_key"])

    def _load_crypto_params(self):        
        cc, pb, sk = \
            self._he_scheme_config_pb.crypto_context, \
            self._he_scheme_config_pb.public_key, \
            self._he_scheme_config_pb.private_key

        # We only load crypto params if they are defined.
        if self._he_scheme_config_pb.as_files:
            if (cc): self._he_scheme.load_crypto_context_from_file(cc)
            if (pb): self._he_scheme.load_public_key_from_file(pb)
            if (sk): self._he_scheme.load_private_key_from_file(sk)
        else:
            if (cc): self._he_scheme.load_crypto_context(cc)
            if (pb): self._he_scheme.load_public_key(pb)
            if (sk): self._he_scheme.load_private_key(sk)
