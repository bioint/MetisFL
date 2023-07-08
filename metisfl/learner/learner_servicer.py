import threading

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from metisfl import config
from metisfl.grpc.grpc_services import GRPCServerMaxMsgLength
from metisfl.proto import learner_pb2_grpc
from metisfl.proto.metis_pb2 import ServerEntity
from metisfl.utils.metis_logger import MetisLogger
from metisfl.utils.proto_messages_factory import (LearnerServiceProtoMessages,
                                                  ServiceCommonProtoMessages)

from .grpc_controller_client import GRPCControllerClient
from .learner_executor import LearnerExecutor


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self,
                 learner_executor: LearnerExecutor,
                 controller_server_entity_pb: ServerEntity,
                 learner_server_entity_pb: ServerEntity,
                 dataset_metadata: dict,
                 servicer_workers=10):
        self._learner_executor = learner_executor
        self._servicer_workers = servicer_workers
        self.__community_models_received = 0
        self.__model_evaluation_requests = 0
        # event to stop serving inbound requests
        self.__not_serving_event = threading.Event()
        # event to stop all grpc related tasks
        self.__shutdown_event = threading.Event()

        # @stripeli: any reason why this was not in __init__?
        self.__grpc_server = GRPCServerMaxMsgLength(
            max_workers=self._servicer_workers,
            server_entity=learner_server_entity_pb,
        )

        self._learner_controller_client = GRPCControllerClient(
            controller_server_entity=controller_server_entity_pb,
            learner_server_entity=learner_server_entity_pb,
            dataset_metadata=dataset_metadata
        )

    def init_servicer(self):
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(
            self, self.__grpc_server.server)
        self.__grpc_server.server.start()
        self._learner_controller_client.join_federation()
        self._log_init_learner()
        self.__shutdown_event.wait()
        self.__grpc_server.server.stop(None)

    def EvaluateModel(self, request, context):
        if not self._is_serving(context):
            return LearnerServiceProtoMessages \
                .construct_evaluate_model_response_pb()
        self._log_evaluation_task_receive()
        self.__model_evaluation_requests += 1  # @stripeli where is this used?
        
        # Unpack these from the request as they are repeated proto fields
        # and can't be pickled
        evaluation_dataset_pb = [d for d in request.evaluation_dataset]
        metric_pb = [m for m in request.metrics.metric]
        
        model_evaluations_pb = self._learner_executor.run_evaluation_task(
            model_pb=request.model,
            batch_size=request.batch_size,
            evaluation_dataset_pb=evaluation_dataset_pb,
            metrics_pb=metric_pb, 
            cancel_running=False,
            block=True,
            verbose=True)
        
        evaluate_model_response_pb = LearnerServiceProtoMessages \
            .construct_evaluate_model_response_pb(model_evaluations_pb)
        return evaluate_model_response_pb

    def GetServicesHealthStatus(self, request, context):
        if not self._is_serving(context):
            return ServiceCommonProtoMessages \
                .construct_get_services_health_status_request_pb()
        self._log_health_check_receive()
        services_status = {"server": self.__grpc_server.server is not None}
        return ServiceCommonProtoMessages \
            .construct_get_services_health_status_response_pb(services_status=services_status)

    def RunTask(self, request, context):
        if self._is_serving(context) is False:
            return LearnerServiceProtoMessages \
                .construct_run_task_response_pb()
        self._log_training_task_receive()
        self.__community_models_received += 1
        is_task_submitted = self._learner_executor.run_learning_task(
            callback=self._learner_controller_client.mark_task_completed,
            learning_task_pb=request.task,
            model_pb=request.federated_model.model,
            hyperparameters_pb=request.hyperparameters,
            cancel_running_tasks=True,
            block=False,
            verbose=True)
        return self._get_task_response_pb(is_task_submitted)

    def ShutDown(self, request, context):
        self._log_shutdown()
        self.__not_serving_event.set()
        self._learner_executor.shutdown(
            CANCEL_RUNNING=config.CANCEL_RUNNING_ON_SHUTDOWN)
        self._learner_controller_client.leave_federation()
        self.__shutdown_event.set()
        return self._get_shutdown_response_pb()

    def _get_task_response_pb(self, is_task_submitted):
        ack_pb = ServiceCommonProtoMessages.construct_ack_pb(
            status=is_task_submitted,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        return LearnerServiceProtoMessages.construct_run_task_response_pb(ack_pb)

    def _get_shutdown_response_pb(self):
        ack_pb = ServiceCommonProtoMessages.construct_ack_pb(
            status=True,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        return ServiceCommonProtoMessages.construct_shutdown_response_pb(ack_pb)

    def _is_serving(self, context):
        if self.__not_serving_event.is_set():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True

    def _log_init_learner(self):
        MetisLogger.info("Initialized Learner Servicer {}".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_evaluation_task_receive(self):
        MetisLogger.info("Learner Servicer {} received model evaluation task.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_health_check_receive(self):
        MetisLogger.info("Learner Servicer {} received a health status request.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_training_task_receive(self):
        MetisLogger.info("Learner Servicer {} received local training task.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_shutdown(self):
        MetisLogger.info("Learner Servicer {} received shutdown request.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))
