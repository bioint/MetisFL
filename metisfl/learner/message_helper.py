
"""This module contains the MessageHelper class of the learner,
    which is used to convert the weights of the model to a Proto object and vice versa."""

from typing import List, Optional

import numpy as np

from metisfl.encryption.scheme import EncryptionScheme
from metisfl.proto import model_pb2


class MessageHelper:

    def __init__(
        self,
        scheme: Optional[EncryptionScheme] = None
    ) -> None:
        """Initializes the MessageHelper object.

        Parameters
        ----------
        scheme : Optional[EncryptionScheme], optional
            The encryption scheme to be used, by default None (no encryption).
        """
        self.scheme = scheme

    def weights_to_model_proto(self, weights: List[np.ndarray]) -> model_pb2.Model:
        """Converts the weights of the model to a Proto object.

        Parameters
        ----------
        weights : List[np.ndarray]
            The weights of the model.

        Returns
        -------
        model_pb2.Model
            The Proto object with the model.
        """

        model = model_pb2.Model()
        for weight in weights:
            tensor = model.tensors.add()
            tensor.length = weight.size
            tensor.dimensions.extend(weight.shape)
            weight = weight.astype(np.float64)

            if self.scheme is not None:
                model.encrypted = True
                tensor.value = self.scheme.encrypt(weight.flatten())
            else:
                model.encrypted = False
                tensor.value = weight.flatten().tobytes()

        return model

    def model_proto_to_weights(self, model: model_pb2.Model) -> List[np.ndarray]:
        """Converts the Proto object with the model to the weights of the model.

        Parameters
        ----------
        model : model_pb2.Model
            The Proto object with the model.

        Returns
        -------
        List[np.ndarray]
            The weights of the model.
        """

        if model.encrypted:
            if self.scheme is None:
                raise ValueError(
                    "Model is encrypted but no encryption scheme was provided")

        weights = []
        for tensor in model.tensors:
            if model.encrypted:
                decrypted = self.scheme.decrypt(tensor.value, tensor.length)
                weights.append(
                    np.array(
                        decrypted,
                        dtype=np.float64
                    ).reshape(tensor.dimensions)
                )
            else:
                weights.append(
                    np.frombuffer(
                        buffer=tensor.value,
                        dtype=np.float64,
                        count=tensor.length
                    ).reshape(tensor.dimensions)
                )
        return weights
