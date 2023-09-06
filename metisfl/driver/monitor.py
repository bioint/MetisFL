
import datetime
import time
from typing import Dict, Optional, Union

import numpy as np
from google.protobuf.json_format import MessageToDict
from loguru import logger

from metisfl.common.types import TerminationSingals
from metisfl.driver.controller_client import GRPCControllerClient
from metisfl.proto import controller_pb2


class FederationMonitor:

    """A monitoring service for the federation."""

    def __init__(
        self,
        termination_signals: TerminationSingals,
        controller_client: GRPCControllerClient,
        log_request_interval_secs: Optional[int] = 15
    ):
        """Initializes the service monitor for the federation.

        Parameters
        ----------
        termination_signals : TerminationSingals
            The termination signals for the federation. When any of the signals is reached, the training is terminated.
        controller_client : GRPCControllerClient
            A gRPC client used from the driver to communicate with the controller.
        log_request_interval_secs : Optional[int], optional
            The interval in seconds to request statistics from the Controller, by default 15
        """

        self._controller_client = controller_client
        self._signals = termination_signals
        self._log_request_interval_secs = log_request_interval_secs
        self._logs = None

    def monitor_federation(self) -> Union[Dict, None]:
        """Monitors the federation. 

        The controller and learners are terminated when any of the termination signals is reached,
        and the collected statistics are returned.

        Returns
        -------
        Dict
            The collected statistics from the federation.
        """

        st = datetime.datetime.now()
        terminate = False

        while not terminate:
            time.sleep(self._log_request_interval_secs)
            logger.info("Requesting logs from controller ...")

            self._get_logs()

            terminate = self._reached_federation_rounds() or \
                self._reached_evaluation_score() or \
                self._reached_execution_time(st)

        return self._logs

    def _reached_federation_rounds(self) -> bool:
        """Checks if the federation has reached the maximum number of rounds."""

        if not self._signals.federation_rounds or "global_iteration" not in self._logs:
            return False

        if self._logs["global_iteration"] >= self._signals.federation_rounds:
            logger.info(
                "Exceeded federation rounds cutoff point. Exiting ...")
            return True

        return False

    def _reached_evaluation_score(self) -> bool:
        """Checks if the federation has reached the maximum evaluation score."""

        metric = self._signals.evaluation_metric
        cutoff_score = self._signals.evaluation_metric_cutoff_score

        if not metric or not cutoff_score:
            return False

        eval_metric = {}  # learner_id -> eval_metric
        timestamps = {}  # learner_id -> timestamp

        tasks = self._logs["tasks"]
        eval_results = self._logs["evaluation_results"]

        for task in tasks:
            task_id = task["id"]
            learner_id = task["learner_id"]

            if task_id not in eval_results or\
                    metric not in eval_results[task_id]["metrics"]:
                continue

            if learner_id not in eval_metric or \
                    timestamps[learner_id] < task.completed_at:
                eval_metric[learner_id] = eval_results[task_id].metrics[metric]
                timestamps[learner_id] = task.completed_at

        if np.mean(list(eval_metric.values())) > cutoff_score:
            logger.info(
                f"Exceeded evaluation score cutoff point. Exiting ...")
            return True

        return False

    def _reached_execution_time(self, st) -> bool:
        """Checks if the federation has exceeded the maximum execution time."""

        et = datetime.datetime.now()
        diff_mins = (et - st).seconds / 60
        cutoff_mins = self._signals.execution_cutoff_time_mins

        if cutoff_mins and diff_mins >= cutoff_mins:
            logger.info(
                "Exceeded execution time cutoff minutes. Exiting ...")
            return True

        return False

    def _get_logs(self) -> controller_pb2.Logs:
        """Collects statistics from the federation."""

        self._logs = self._controller_client.get_logs()

        if self._logs is None:
            raise Exception("Failed to get logs from controller.")

        def msg_convert(msg):
            return MessageToDict(msg, preserving_proto_field_name=True)

        def proto_map_to_dict(proto_map):
            return {k: msg_convert(v) for k, v in proto_map.items()}

        logs = {
            "tasks": [msg_convert(task) for task in self._logs.tasks],
            "train_results": proto_map_to_dict(self._logs.train_results),
            "evaluation_results": proto_map_to_dict(self._logs.evaluation_results),
            "model_metadata":  proto_map_to_dict(self._logs.model_metadata),
        }

        if self._logs.HasField("global_iteration"):
            logs["global_iteration"] = self._logs.global_iteration

        self._logs = logs

        return logs
