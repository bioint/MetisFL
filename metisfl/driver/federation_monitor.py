
import datetime
import time
from typing import Dict

from google.protobuf.json_format import MessageToDict

from .controller_client import GRPCControllerClient
from ..common.types import TerminationSingals
from ..common.logger import MetisLogger


class FederationMonitor:

    """A monitoring service for the federation."""

    def __init__(
        self,
        termination_signals: TerminationSingals,
        controller_client: GRPCControllerClient,
        is_async: bool
    ):
        """Initializes the service monitor for the federation.

        Parameters
        ----------
        termination_signals : TerminationSingals
            The termination signals for the federation. When any of the signals is reached, the training is terminated.
        controller_client : GRPCControllerClient
            A gRPC client used from the driver to communicate with the controller.
        is_async : bool
            Whether the communication protocol is asynchronous.
        """

        self._controller_client = controller_client
        self._signals = termination_signals
        self._is_async = is_async
        self._statistics = None

    def monitor_federation(self, request_every_secs=10) -> Dict:
        """Monitors the federation. 

        The controller and learners are terminated when any of the termination signals is reached,
        and the collected statistics are returned.

        Parameters
        ----------
        request_every_secs : int, optional
            The interval in seconds to request statistics from the Controller, by default 10

        Returns
        -------
        Dict
            The collected statistics from the federation.
        """

        st = datetime.datetime.now()
        terminate = False

        while not terminate:
            time.sleep(request_every_secs)
            self._collect_statistics()

            terminate = self._reached_federation_rounds() or \
                self._reached_evaluation_score() or \
                self._reached_execution_time(st)

        return self._statistics

    def _reached_federation_rounds(self) -> bool:
        """Checks if the federation has reached the maximum number of rounds."""

        if not self._signals.federation_rounds or self._is_async:
            return False

        metadata = self._statistics["metadata"]

        if metadata:
            current_global_iteration = max(
                [m.global_iteration for m in metadata])

            if current_global_iteration > self._signals.federation_rounds:
                MetisLogger.info(
                    "Exceeded federation rounds cutoff point. Exiting ...")
                return True
        return False

    def _reached_evaluation_score(self) -> bool:
        """Checks if the federation has reached the maximum evaluation score."""

        metric_cutoff_score = self._signals.evaluation_metric_cutoff_score

        if not metric_cutoff_score:
            return False

        commmunity_evaluation = [
            x for x in self._statistics["community_evaluation"]]

        for res in commmunity_evaluation:
            scores = []
            for _, evaluations in res.evaluations.items():
                evaluation_metric = evaluations.evaluation.metric_values

                if evaluation_metric and evaluation_metric in evaluations.evaluation.metric_values:
                    test_score = evaluations.evaluation.metric_values[evaluation_metric]
                    scores.append(float(test_score))

            if scores:
                mean_score = sum(scores) / len(scores)
                if mean_score >= metric_cutoff_score:
                    MetisLogger.info(
                        "Exceeded evaluation metric cutoff score. Exiting...")
                    return True
        return False

    def _reached_execution_time(self, st) -> bool:
        """Checks if the federation has exceeded the maximum execution time."""

        et = datetime.datetime.now()
        diff_mins = (et - st).seconds / 60
        cutoff_mins = self._signals.execution_cutoff_time_mins

        if cutoff_mins and diff_mins >= cutoff_mins:
            MetisLogger.info(
                "Exceeded execution time cutoff minutes. Exiting ...")
            return True

        return False

    def _collect_statistics(self) -> None:
        """Collects statistics from the federation."""

        statistics_pb = self._controller_client.get_statistics(
            local_task_backtracks=-1,
            metadata_backtracks=0,
            community_evaluation_backtracks=-1,
        )

        def msg_to_dict_fn(x): return MessageToDict(
            x, preserving_proto_field_name=True)

        statistics = {}
        # FIXME:
        statistics["learners"] = msg_to_dict_fn(statistics_pb.learners)
        statistics["metadata"] = msg_to_dict_fn(statistics_pb.metadata)
        statistics["learners_task"] = msg_to_dict_fn(
            statistics_pb.learners_task)
        statistics["community_evaluation"] = msg_to_dict_fn(
            statistics_pb.community_model_lineage)

        self._statistics = statistics
