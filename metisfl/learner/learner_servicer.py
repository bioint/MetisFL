

import threading

import grpc

import config

from ..grpc.server import get_server
from ..proto import learner_pb2, learner_pb2_grpc, service_common_pb2
from ..utils.fedenv import ServerParams
from ..utils.logger import MetisLogger
from .controller_client import GRPCControllerClient
from .learner_executor import LearnerExecutor


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(
        self,
        learner_executor: LearnerExecutor,
        controller_server_params: ServerParams,
        learner_server_params: ServerParams,
    ):
        """Initializes the LearnerServicer."""
        
        self._learner_executor = learner_executor
        self._not_serving_event = threading.Event()
        self._shutdown_event = threading.Event()
        
        self._client = GRPCControllerClient(
            server_params=controller_server_params,
            learner_id_fp=config.get_auth_token_fp(learner_server_params.port),
            auth_token_fp=config.get_auth_token_fp(learner_server_params.port),
        )
        
        self._server = get_server(
            server_params=learner_server_params,
            servicer_and_add_fn=(
                self, learner_pb2_grpc.add_LearnerServiceServicer_to_server),
        )

    def start(self):
        self._server.start()
        self._client.join_federation() # FIXME: requires num_training_examples
        self._log_init_learner()
        self._shutdown_event.wait()
        self._server.stop(None)

    def EvaluateModel(self, request, context):
        if not self._is_serving(context):
            return learner_pb2.EvaluateModelResponse(evaluations=None)

        model_evaluations_pb = self._learner_executor.run_evaluation_task(
            model_pb=request.model,
            batch_size=request.batch_size,
            cancel_running_tasks=False,
            block=True,
            verbose=True)

        return learner_pb2.EvaluateModelResponse(evaluations=model_evaluations_pb)

    def GetHealthStatus(self, request, context):
        if not self._is_serving(context):
            return service_common_pb2.GetServicesHealthStatusRequest()

        self._log_health_check_receive()

        services_status = {"server": self._server is not None}

        return service_common_pb2.GetServicesHealthStatusResponse(services_status=services_status)

    def RunTask(self, request, context):
        if self._is_serving(context) is False:
            return learner_pb2.RunTaskResponse(ack=None)

        self._log_training_task_receive()

        is_task_submitted = self._learner_executor.run_learning_task(
            callback=self._client.mark_task_completed,
            learning_task_pb=request.task,
            model_pb=request.federated_model.model,
            hyperparameters_pb=request.hyperparameters,
            cancel_running_tasks=True,
            block=False,
            verbose=True)

        return _get_task_response_pb(is_task_submitted)

    def ShutDown(self):
        self._log_shutdown()
        self._not_serving_event.set()
        self._learner_executor.shutdown(
            CANCEL_RUNNING=config.CANCEL_RUNNING_ON_SHUTDOWN)
        self._client.leave_federation()
        self._shutdown_event.set()
        return _get_shutdown_response_pb()

    def _is_serving(self, context):
        if self._not_serving_event.is_set():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True

    def _log_init_learner(self):
        MetisLogger.info("Initialized Learner GRPC Server {}".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_evaluation_task_receive(self):
        MetisLogger.info("Learner Server {} received model evaluation task.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_health_check_receive(self):
        MetisLogger.info("Learner Server {} received a health status request.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_training_task_receive(self):
        MetisLogger.info("Learner Server {} received local training task.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))

    def _log_shutdown(self):
        MetisLogger.info("Learner Server {} received shutdown request.".format(
            self.__grpc_server.grpc_endpoint.listening_endpoint))


def _get_task_response_pb(is_task_submitted):
    ack_pb = service_common_pb2.Ack(
        status=is_task_submitted,
        timestamp=Timestamp().GetCurrentTime(),
        message=None
    )
    return learner_pb2.RunTaskResponse(ack=ack_pb)


def _get_shutdown_response_pb():
    ack_pb = service_common_pb2.Ack(
        status=True,
        timestamp=Timestamp().GetCurrentTime(),
        message=None
    )
    return service_common_pb2.ShutDownResponse(ack=ack_pb)
