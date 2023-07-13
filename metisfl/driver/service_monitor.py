import datetime
import time

from google.protobuf.json_format import MessageToDict

from metisfl.utils.fedenv import FederationEnvironment
from metisfl.utils.metis_logger import MetisLogger

from .controller_client import GRPCControllerClient


class ServiceMonitor:

    def __init__(self,
                 federation_environment: FederationEnvironment,
                 driver_controller_grpc_client: GRPCControllerClient):
        self._driver_controller_grpc_client = driver_controller_grpc_client
        self._federation_environment = federation_environment
        self._federation_rounds_cutoff = self._federation_environment.federation_rounds
        self._communication_protocol = self._federation_environment.communication_protocol
        self._execution_time_cutoff_mins = self._federation_environment.execution_time_cutoff_mins
        self._metric_cutoff_score = self._federation_environment.metric_cutoff_score
        self._evaluation_metric = self._federation_environment.evaluation_metric
        self._federation_statistics = dict()

    def monitor_federation(self, request_every_secs=10):
        self._monitor_termination_signals(
            request_every_secs=request_every_secs)

    def get_federation_statistics(self):
        return self._federation_statistics

    def _monitor_termination_signals(self, request_every_secs=10):
        # measuring elapsed wall-clock time
        st = datetime.datetime.now()
        terminate = False
        while not terminate:
            # ping controller for latest execution stats
            time.sleep(request_every_secs)

            terminate = self._reached_federation_rounds() or \
                self._reached_evaluation_score() or \
                self._reached_execution_time(st)

    def _reached_federation_rounds(self) -> bool:
        metadata_pb = self._driver_controller_grpc_client \
            .get_runtime_metadata(num_backtracks=0).metadata
        if not self._is_async():
            if self._federation_rounds_cutoff and len(metadata_pb) > 0:
                current_global_iteration = max(
                    [m.global_iteration for m in metadata_pb])
                if current_global_iteration > self._federation_rounds_cutoff:
                    MetisLogger.info(
                        "Exceeded federation rounds cutoff point. Exiting ...")
                    return True
        return False

    def _is_async(self) -> bool:
        return "async" in self._communication_protocol.lower()

    def _reached_evaluation_score(self) -> bool:
        community_results = self._driver_controller_grpc_client \
            .get_community_model_evaluation_lineage(-1)

        # Need to materialize the iterator in order to get all community results.
        community_results = [x for x in community_results.community_evaluation]

        for res in community_results:
            test_set_scores = []
            # Since we evaluate the community model across all learners,
            # we need to measure the average performance across the test sets.
            # FIXME(@stripeli) : what if there is no test set?
            for learner_id, evaluations in res.evaluations.items():
                if self._evaluation_metric and self._evaluation_metric in evaluations.test_evaluation.metric_values:
                    test_score = evaluations.test_evaluation.metric_values[self._evaluation_metric]
                    test_set_scores.append(float(test_score))
            if test_set_scores:
                mean_test_score = sum(test_set_scores) / len(test_set_scores)
                if mean_test_score >= self._metric_cutoff_score:
                    MetisLogger.info(
                        "Exceeded evaluation metric cutoff score. Exiting ...")
                    return True
        return False

    def _reached_execution_time(self, st) -> bool:
        et = datetime.datetime.now()
        diff_mins = (et - st).seconds / 60
        if self._execution_time_cutoff_mins and diff_mins > self._execution_time_cutoff_mins:
            MetisLogger.info(
                "Exceeded execution time cutoff minutes. Exiting ...")
            return True
        return False

    def collect_local_statistics(self) -> None:
        learners_pb = self._driver_controller_grpc_client.get_participating_learners()
        learners_collection = learners_pb.learner
        learners_id = [learner.id for learner in learners_collection]
        learners_descriptors_dict = MessageToDict(learners_pb,
                                                  preserving_proto_field_name=True)
        learners_results = self._driver_controller_grpc_client \
            .get_local_task_lineage(-1, learners_id)
        learners_results_dict = MessageToDict(learners_results,
                                              preserving_proto_field_name=True)
        self._federation_statistics["learners_descriptor"] = learners_descriptors_dict
        self._federation_statistics["learners_models_results"] = learners_results_dict

    def collect_global_statistics(self) -> None:
        runtime_metadata_pb = self._driver_controller_grpc_client \
            .get_runtime_metadata(num_backtracks=0)
        runtime_metadata_dict = MessageToDict(runtime_metadata_pb,
                                              preserving_proto_field_name=True)
        community_results = self._driver_controller_grpc_client \
            .get_community_model_evaluation_lineage(-1)
        community_results_dict = MessageToDict(community_results,
                                               preserving_proto_field_name=True)
        self._federation_statistics["federation_runtime_metadata"] = runtime_metadata_dict
        self._federation_statistics["community_model_results"] = community_results_dict

    def get_statistics(self) -> dict:
        return self._federation_statistics
