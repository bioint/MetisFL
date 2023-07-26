import abc

from typing import Any, List
from metisfl.proto import metis_pb2
from metisfl.utils.logger import MetisLogger


class EncryptionScheme(object):

    def __init__(self, init_crypto_params=False) -> None:
        self.init_crypto_params = init_crypto_params

    def from_proto(self, 
                   encryption_config: metis_pb2.EncryptionConfig = None):
        if encryption_config:
            # Import happens at the function level to avoid circular imports.
            if encryption_config.HasField("he_scheme"):
                from metisfl.encryption.homomorphic import Homomorphic
                return Homomorphic(encryption_scheme.he_scheme, self.init_crypto_params)
            elif encryption_config.HasField("masking_scheme"):
                from metisfl.encryption.masking import Masking
                return Masking(encryption_config.masking_scheme, self.init_crypto_params)
            else:
                MetisLogger.fatal(
                    "Not a supported encryption scheme: {}".format(
                    encryption_config))
        else:
            return None   

    @abc.abstractmethod
    def decrypt_data(self, ciphertext: str, num_elems: int) -> List[Any]:
        pass

    @abc.abstractmethod
    def encrypt_data(self, values) -> Any:
        pass

    @abc.abstractmethod
    def to_proto(self) -> metis_pb2.EncryptionConfig:
        pass
