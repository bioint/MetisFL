
""" MetisFL Asynchronous Scheduler """


from typing import List

from metisfl.server.scheduling.scheduler import Scheduler


class AsynchronousScheduler(Scheduler):

    global_iteration: int = 0

    def schedule(
        self,
        learner_id: str,
    ) -> List[str]:
        """Schedule the next batch of learners, asynchronously.

        Parameters
        ----------
        learner_id : str
            The ID of the learner to schedule.
        num_active_learners : int
            The number of active learners.

        Returns
        -------
        List[str]
            The IDs of the learners to schedule.
        """

        self.global_iteration += 1

        return [learner_id]

    def __str__(self) -> str:
        return "AsynchronousScheduler"
