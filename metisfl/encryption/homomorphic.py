
"""MetisFL Homomorphic Encryption Module using Palisade."""

import numpy as np
from metisfl.encryption import fhe
from metisfl.encryption.scheme import EncryptionScheme


class HomomorphicEncryption(EncryptionScheme):

    """Homomorphic Encryption class using Palisade. Wraps the C++ implementation of Palisade."""

    def __init__(
        self,
        batch_size: int,
        scaling_factor_bits: int,
        crypto_context_path: str,
        public_key_path: str,
        private_key_path: str,
    ):
        """Initializes the CKKS Homomorphic Encryption scheme. 

        Parameters
        ----------
        batch_size : int
            The batch size of the encryption scheme.
        scaling_factor_bits : int
            The number of bits to use for the scaling factor.
        crypto_context_path : str
            The path to the crypto context file.
        public_key_path : str
            The path to the public key file.
        private_key_path : str
            The path to the private key file.

        """
        self.ckks = fhe.CKKS(
            batch_size, scaling_factor_bits, crypto_context_path, public_key_path, private_key_path
        )

    def decrypt(self, value: bytes, length: int) -> np.ndarray:
        """Decrypts the value.

        Parameters
        ----------
        value : bytes
            The value to decrypt as bytes.
        length : int
            The length of the value.

        Returns
        -------
        np.ndarray
            The decrypted value as a numpy array.
        """

        return self.ckks.decrypt(value, length)

    def encrypt(self, arr: np.ndarray) -> bytes:
        """Encrypts the array.

        Parameters
        ----------
        arr : np.ndarray
            The array to encrypt.

        Returns
        -------
        bytes
            The encrypted array as bytes.
        """
        
        arr = np.array(arr).flatten()

        return self.ckks.encrypt(arr)
