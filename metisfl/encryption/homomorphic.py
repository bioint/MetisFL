
"""MetisFL Homomorphic Encryption Module using Palisade."""

from metisfl.encryption import fhe
from ..proto import model_pb2


class HomomorphicEncryption(object):

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
        crypto_context_path : str, optional
            The path to the crypto context file.
        public_key_path : str, optional
            The path to the public key file.
        private_key_path : str, optional
            The path to the private key file.

        """
        # TODO: Make it easier to load the crypto context and keys.
        self._he_scheme = fhe.CKKS(batch_size, scaling_factor_bits)
        self._he_scheme.load_crypto_context_from_file(crypto_context_path)
        self._he_scheme.load_public_key_from_file(public_key_path)
        self._he_scheme.load_private_key_from_file(private_key_path)

    def decrypt(self, model: model_pb2.Model) -> model_pb2.Model:
        """Decrypts the model in place (if encrypted).

        Parameters
        ----------
        model : model_pb2.Model
            The model to decrypt.

        Returns
        -------
        model_pb2.Model
            The decrypted model.
        """

        for tensor in model.tensors:
            if tensor.encryped:
                decoded_value = self._he_scheme.decrypt(
                    tensor.value, tensor.length)
                tensor.value = decoded_value
                tensor.encrypted = False

        return model

    def encrypt(self, model: model_pb2.Model) -> model_pb2.Model:
        """Encrypts the model in place (if not encrypted).

        Parameters
        ----------
        model : model_pb2.Model
            The model to encrypt.

        Returns
        -------
        model_pb2.Model
            The encrypted model.
        """
        # FIXME:
        for tensor in model.tensors:
            if not tensor.encrypted:
                tensor.value = self._he_scheme.encrypt(tensor.value)
                tensor.encrypted = True

        return model
