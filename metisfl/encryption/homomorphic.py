import os

from typing import Dict

from metisfl.utils.metis_logger import MetisLogger
from metisfl.encryption import fhe
from metisfl.proto import metis_pb2


class Homomorphic(object):

    def __init__(self, he_scheme_pb: metis_pb2.HESchemeConfig):
        assert isinstance(
            he_scheme_pb, metis_pb2.HESchemeConfig), "Not a valid HE scheme protobuf."
        
        self._he_scheme = None        
        if he_scheme_pb.HasField("ckks_scheme_config"):
            self._he_scheme = self._construct_ckks_scheme(he_scheme_pb)
        elif he_scheme_pb.HasField("empty_scheme_config"):
            self._he_scheme = metis_pb2.EmptySchemeConfig()
        else:
            raise MetisLogger.fatal(
                "Not a supported HE scheme config. Received: {}".format(he_scheme_pb))

    @staticmethod
    def from_proto(he_scheme_pb: metis_pb2.HESchemeConfig):
        return Homomorphic(he_scheme_pb)
    
    @staticmethod
    def generate_crypto_params_ckks(crypto_dir, batch_size, scaling_factor_bits) -> Dict:
        assert batch_size is not None, "Batch size cannot be empty."
        assert scaling_factor_bits is not None, "Scaling factor bits cannot be empty."
        ckks_scheme = fhe.CKKS(batch_size, scaling_factor_bits)
        if not os.path.exists(crypto_dir):
            MetisLogger.info(
                "Creating crypto parameters directory: {}".format(crypto_dir))
            os.makedirs(crypto_dir)
        ckks_scheme.gen_crypto_context_and_keys(crypto_dir)
        crypto_params_files = ckks_scheme.get_crypto_params_files()
        MetisLogger.info("Crypto parameters files:")
        for param, filename in crypto_params_files.items():
            MetisLogger.info("\t {}:{}".format(param, filename))
        return crypto_params_files

    def decrypt_data(self, ciphertext: str, num_elems: int):
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:
            return self._he_scheme.decrypt(ciphertext, num_elems)

    def encrypt_data(self, values):
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:                
            return self._he_scheme.encrypt(values)

    def _construct_ckks_scheme(self, he_scheme_pb: metis_pb2.HESchemeConfig):
        ckks_scheme = fhe.CKKS(he_scheme_pb.ckks_scheme_config.batch_size,
                               he_scheme_pb.ckks_scheme_config.scaling_factor_bits)
        if he_scheme_pb.crypto_context_file:
            ckks_scheme.load_crypto_context_from_file(he_scheme_pb.crypto_context_file)
        if he_scheme_pb.public_key_file:
            ckks_scheme.load_public_key_from_file(he_scheme_pb.public_key_file)
        if he_scheme_pb.private_key_file:
            ckks_scheme.load_private_key_from_file(he_scheme_pb.private_key_file)
        return ckks_scheme
