

from typing import List


class ScheduledCardinality:

    def select(
        self,
        scheduled_learners: List[str],
        active_learners: List[str],
    ) -> List[str]:
        """ A subset cardinality selector that picks the models that need to be
            considered during aggregation based on the cardinality of the learners
            subset (scheduled) collection.

        Parameters
        ----------
        scheduled_learners : List[str]
            The list of learners that are scheduled.
        active_learners : List[str]
            The list of learners that are active.

        Returns
        -------
        List[str]
            The list of learners selected from the collection of scheduled learners.
        """

        if len(scheduled_learners) < 2:
            return active_learners
        else:
            return scheduled_learners

    def __str__(self):
        return "ScheduledCardinality"
