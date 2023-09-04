
from abc import ABC, abstractmethod
from typing import List, Optional


class Scheduler(ABC):

    @abstractmethod
    def schedule(
        self,
        learner_id: str,
    ) -> List[str]:
        """Schedule the next batch of learners.

        Parameters
        ----------
        learner_id : str
            The ID of the learner to schedule.

        Returns
        -------
        List[str]
            The IDs of the learners to schedule.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
