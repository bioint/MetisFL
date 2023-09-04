from typing import Dict, List, Tuple
from metisfl.common.logger import MetisLogger
import metisfl.proto.model_pb2 as model_pb2
from metisfl.server.store.model_store import ModelStore


class HashMapModelStore(ModelStore):

    lineage_length: int = None
    store_cache: Dict[str, List[model_pb2.Model]] = {}

    def __init__(self, lineage_length: int):
        """Initializes the HashMapModelStore

        Parameters
        ----------
        lineage_length : int
            The lineage length of the model store, i.e., the maximum number of
            models that can be stored for each learner. If 0, then the lineage
            length is infinite.
        """
        self.lineage_length = lineage_length

        MetisLogger.info(
            f"HashMapModelStore initialized with lineage length {lineage_length}")

    def expunge(self):
        self.store_cache = {}

    def erase(self, learner_ids: List[str]) -> None:
        """Erases the models for the given learner ids.

        Parameters
        ----------
        learner_ids : List[str]
            The list of learner ids whose models are to be erased.
        """

        for learner_id in learner_ids:
            if learner_id in self.store_cache:
                self.store_cache.pop(learner_id)

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

        if not learner_id in self.store_cache:
            raise ValueError(f"No models found for learner id {learner_id}")

        return len(self.store_cache[learner_id])

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

        for pair in pairs:
            learner_id, model = pair

            if learner_id not in self.store_cache:
                self.store_cache[learner_id] = []

            self.store_cache[learner_id].append(model)

            if self.lineage_length > 0:
                if len(self.store_cache[learner_id]) > self.lineage_length:
                    self.store_cache[learner_id].pop(0)

    def select(
        self,
        pairs: List[Tuple[str, int]]
    ) -> Dict[str, List[model_pb2.Model]]:
        """Selects given number of models for each learner id.

        Parameters
        ----------
        pairs : List[Tuple[str, int]]
            A list of (learner_id, number of models) pairs.

        Returns
        -------
        Dict[str, List[model_pb2.Model]]
            A dictionary of learner id to list of models.    
        """

        result = {}

        for pair in pairs:
            learner_id, num_models = pair

            if learner_id not in self.store_cache.keys():
                raise ValueError(
                    f"No models found for learner id {learner_id}")

            history_length = len(self.store_cache[learner_id])

            if num_models > history_length:
                # TODO: check if continue is the right thing to do here. How about we return all the models we have?
                MetisLogger.warn(
                    f"Number of models requested ({num_models}) is greater than "
                    f"the number of models available ({history_length}) for "
                    f"learner id {learner_id}")
                continue

            if num_models <= 0:
                num_models = history_length

            result[learner_id] = self.store_cache[learner_id][-num_models:]

        return result
