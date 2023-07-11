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
            raise MetisLogger.fatal("Not a supported HE scheme config. Received: {}".format(he_scheme_pb))

    @staticmethod
    def from_proto(he_scheme_pb: metis_pb2.HESchemeConfig):
        return Homomorphic(he_scheme_pb)

    def decrypt_data(self, ciphertext: str, num_elems: int):
        assert self._he_scheme, "The homomorphic encryption scheme is not initialized."
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:
            return self._he_scheme.decrypt(ciphertext, num_elems)

    def encrypt_data(self, values):
        assert self._he_scheme, "The homomorphic encryption scheme is not initialized."
        if isinstance(self._he_scheme, metis_pb2.EmptySchemeConfig):
            return None
        else:                
            return self._he_scheme.encrypt(values)

    def _load_crypto_params(he_scheme_pb: metis_pb2.HESchemeConfig):
        ckks_scheme = fhe.CKKS(he_scheme_pb.fhe_scheme.batch_size,
                               he_scheme_pb.fhe_scheme.scaling_bits)
        if he_scheme_pb.crypto_context_file:
            ckks_scheme.load_crypto_context_from_file(he_scheme_pb.crypto_context_file)
        if he_scheme_pb.public_key_file:
            ckks_scheme.load_public_key_from_file(he_scheme_pb.public_key_file)
        if he_scheme_pb.private_key_file:
            ckks_scheme.load_private_key_from_file(he_scheme_pb.private_key_file)
        return ckks_scheme
