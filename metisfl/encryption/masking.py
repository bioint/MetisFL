from typing import Dict
from metisfl.encryption.encryption_scheme import EncryptionScheme


class Masking(EncryptionScheme):
    
    def __init__(self, init_crypto_params=False):
        super().__init__(init_crypto_params)
        pass

    def decrypt_data(self, ciphertext: str, num_elems: int):
        pass

    def encrypt_data(self, values):
        pass
    
    def initialize_crypto_params(self) -> Dict:
        pass
