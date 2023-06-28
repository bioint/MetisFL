import datetime
import time

from google.protobuf.json_format import MessageToDict

from metisfl.utils.fedenv_parser import FederationEnvironment
from metisfl.utils.metis_logger import MetisLogger

from .grpc_controller_client import GRPCControllerClient


class FederationMonitor:
    
    def __init__(self, 
                 federation_environment: FederationEnvironment, 
                 driver_controller_grpc_client: GRPCControllerClient):
        self._driver_controller_grpc_client = driver_controller_grpc_client
        self.federation_environment = federation_environment
        self.federation_rounds_cutoff = self.federation_environment.termination_signals.federation_rounds
        self.communication_protocol = self.federation_environment.communication_protocol
        self.execution_time_cutoff_mins = self.federation_environment.termination_signals.execution_time_cutoff_mins
        self.metric_cutoff_score = self.federation_environment.termination_signals.metric_cutoff_score
        self.evaluation_metric = self.federation_environment.evaluation_metric
        self._federation_statistics = dict()

    def monitor_federation(self, request_every_secs=10):
        self.monitor_termination_signals(request_every_secs=request_every_secs)

    def get_federation_statistics(self):
        return self._federation_statistics

    def monitor_termination_signals(self, request_every_secs=10):
        # measuring elapsed wall-clock time
        st = datetime.datetime.now()
        signal_not_reached = True
        while signal_not_reached:
            # ping controller for latest execution stats
            time.sleep(request_every_secs)

            signal_not_reached = self._check_federation_rounds() or \
                                 self._check_evaluation_score() or \
                                 self._check_execution_time(st)
            
    def _check_federation_rounds(self) -> bool:
        metadata_pb = self._driver_controller_grpc_client \
                .get_runtime_metadata(num_backtracks=0).metadata
        if self.communication_protocol.is_synchronous or self.communication_protocol.is_semi_synchronous:
            if self.federation_rounds_cutoff and len(metadata_pb) > 0:
                current_global_iteration = max([m.global_iteration for m in metadata_pb])
                if current_global_iteration > self.federation_rounds_cutoff:
                    MetisLogger.info("Exceeded federation rounds cutoff point. Exiting ...")
                    return False
        return True

    def _check_evaluation_score(self) -> bool:
        community_results = self._driver_controller_grpc_client \
                .get_community_model_evaluation_lineage(-1)
        
        # Need to materialize the iterator in order to get all community results.
        community_results = [x for x in community_results.community_evaluation]

        for res in community_results:
            test_set_scores = []
            # Since we evaluate the community model across all learners,
            # we need to measure the average performance across the test sets.
            for learner_id, evaluations in res.evaluations.items():
                if self.evaluation_metric in evaluations.test_evaluation.metric_values:
                    test_score = evaluations.test_evaluation.metric_values[self.evaluation_metric]
                    test_set_scores.append(float(test_score))
            if test_set_scores:
                mean_test_score = sum(test_set_scores) / len(test_set_scores)
                if mean_test_score >= self.metric_cutoff_score:
                    MetisLogger.info("Exceeded evaluation metric cutoff score. Exiting ...")
                    return False
        return True
                    
    def _check_execution_time(self, st) -> bool:
        et = datetime.datetime.now()
        diff_mins = (et - st).seconds / 60
        if diff_mins > self.execution_time_cutoff_mins:
            MetisLogger.info("Exceeded execution time cutoff minutes. Exiting ...")
            return False
        return True

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
