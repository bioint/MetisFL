
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from metisfl.proto import model_pb2


class ModelStore(ABC):

    @abstractmethod
    def expunge(self):
        """ Erases all the models from the model store. """
        pass

    @abstractmethod
    def erase(self, learner_ids: List[str]) -> None:
        """Erases the models for the given learner ids.

        Parameters
        ----------
        learner_ids : List[str]
            The list of learner ids whose models are to be erased.
        """
        pass

    @abstractmethod
    def get_lineage_length(self, learner_id: str) -> int:
        """Returns the lineage length for the given learner id.

        Parameters
        ----------
        learner_id : str
            The learner id.

        Returns
        -------
        int
            The lineage length for the given learner id.
        """
        pass

    @abstractmethod
    def insert(
        self,
        pairs: List[Tuple[str, model_pb2.Model]]
    ) -> None:
        """Inserts the given models into the store.

        Parameters
        ----------
        pairs : List[Tuple[str, model_pb2.Model]]
            A list of (learner_id, model) pairs.
        """
        pass

    @abstractmethod
    def select(
        self,
        learner_ids: List[Tuple[str, int]]
    ) -> Dict[Tuple[str, model_pb2.Model]]:
        """Selects the given number of models for each learner id. 

        Parameters
        ----------
        learner_ids : List[Tuple[str, int]]
            A list of (learner_id, num_models) pairs.

        Returns
        -------
        Dict[Tuple[str, model_pb2.Model]]
            A dictionary of (learner_id, model) pairs.
        """
        pass
