"""This file contains the abstract class for encryption schemes."""

from abc import ABC, abstractmethod
from typing import Any


class EncryptionScheme(ABC):

    @abstractmethod
    def encrypt(self, data: Any) -> bytes:
        """Encrypts the data and returns it."""
        pass

    @abstractmethod
    def decrypt(self, data: bytes) -> Any:
        """Decrypts the data and returns it."""
        pass
