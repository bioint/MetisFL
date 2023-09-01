
import datetime
import time
from typing import Dict, List

import numpy as np
from google.protobuf.json_format import MessageToDict

from metisfl.common.logger import MetisLogger
from metisfl.common.types import TerminationSingals
from metisfl.driver.controller_client import GRPCControllerClient
from metisfl.proto import controller_pb2, learner_pb2


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
        self._logs = None

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
            self._get_logs()

            terminate = self._reached_federation_rounds() or \
                self._reached_evaluation_score() or \
                self._reached_execution_time(st)

        return self._get_logs_dict()

    def _reached_federation_rounds(self) -> bool:
        """Checks if the federation has reached the maximum number of rounds."""

        if not self._signals.federation_rounds or self._is_async:
            return False

        if self._logs.global_iteration >= self._signals.federation_rounds:
            MetisLogger.info(
                "Exceeded federation rounds cutoff point. Exiting ...")
            return True

        return False

    def _reached_evaluation_score(self) -> bool:
        """Checks if the federation has reached the maximum evaluation score."""

        metric = self._signals.evaluation_metric
        cutoff_score = self._signals.evaluation_metric_cutoff_score

        if not metric or not cutoff_score:
            return False

        tasks: List[learner_pb2.Task] = self._logs["tasks"]
        eval_results: learner_pb2.EvaluationResults = self._logs["evaluation_results"]
        eval_score = {}
        timestamps = {}

        for task in tasks:
            task_id = task.id
            learner_id = task.learner_id
            if metric not in eval_results[learner_id].metrics:
                MetisLogger.warning(
                    f"Metric {metric} not found in evaluation results for learner {learner_id}.")
                continue

            if learner_id not in eval_results or \
                    timestamps[learner_id] < tasks[task_id].completed_at:
                eval_score[learner_id] = eval_results[learner_id].metrics[metric]
                timestamps[learner_id] = tasks[task_id].completed_at
            else:
                if timestamps[learner_id] < tasks[task_id].completed_at:
                    eval_score[learner_id] = eval_results[learner_id].metrics[metric]
                    timestamps[learner_id] = tasks[task_id].completed_at

        if np.mean(list(eval_score.values())) > cutoff_score:
            MetisLogger.info(
                f"Exceeded evaluation score cutoff point. Exiting ...")
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

    def _get_logs(self) -> controller_pb2.Logs:
        """Collects statistics from the federation."""

        self._logs = self._controller_client.get_logs()

        return self._logs

    def _get_logs_dict(self) -> None:
        """Returns the collected statistics from the federation."""

        def msg_convert(msg):
            return MessageToDict(msg, preserving_proto_field_name=True)

        def proto_map_to_dict(proto_map):
            return {k: msg_convert(v) for k, v in proto_map.items()}

        logs = {
            "global_iteration": self._logs.global_iteration,
            "tasks": [msg_convert(task) for task in self._logs.tasks],
            "train_results": proto_map_to_dict(self._logs.train_results),
            "evaluation_results": proto_map_to_dict(self._logs.evaluation_results),
            "model_metadata":  proto_map_to_dict(self._logs.model_metadata),
        }

        return logs
