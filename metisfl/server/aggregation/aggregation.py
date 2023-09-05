
from abc import ABC, abstractmethod
from typing import List, Tuple

from metisfl.proto import model_pb2


class Aggregator(ABC):
    name: str = None

    @abstractmethod
    def aggregate(self, pairs: List[List[Tuple[model_pb2.Model, float]]]) -> model_pb2.Model:
        """Aggregates the models.

        Parameters
        ----------
        pairs : List[List[Tuple[model_pb2.Model, float]]]
            The models to be aggregated. The first dimension is the learners, and the second dimension is the (model, scaling_factor) pairs.

        Returns
        -------
        Model
            The aggregated model.
        """
        pass

    @abstractmethod
    def required_lineage_length(self) -> int:
        """Returns the required lineage length.

        Returns
        -------
        int
            The required lineage length.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the aggregator."""
        pass
