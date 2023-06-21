import grpc
import threading

import metisfl.utils.proto_messages_factory as proto_factory

from google.protobuf.timestamp_pb2 import Timestamp
from metisfl.utils.grpc_services import GRPCServerMaxMsgLength
from metisfl.proto import learner_pb2_grpc
from metisfl.proto.metis_pb2 import ServerEntity
from metisfl.learner.learner import Learner
from metisfl.utils.metis_logger import MetisLogger


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self, learner: Learner, learner_server_entity: ServerEntity,
                 servicer_workers=10, *args, **kwargs):
        self.learner = learner
        self.learner_server_entity = learner_server_entity
        self.servicer_workers = servicer_workers
        self.__community_models_received = 0
        self.__model_evaluation_requests = 0
        self.__not_serving_event = threading.Event()  # event to stop serving inbound requests
        self.__shutdown_event = threading.Event()  # event to stop all grpc related tasks
        self.__server = None

    def init_servicer(self):
        self.__server = GRPCServerMaxMsgLength(max_workers=self.servicer_workers).server
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(self, self.__server)
        self.__server.add_insecure_port(self.learner.host_port_identifier())
        self.__server.start()
        MetisLogger.info("Initialized Learner Servicer {}".format(
            self.learner.host_port_identifier()))
        # Learner asks controller to join the federation.
        self.learner.join_federation()

    def wait_servicer(self):
        self.__shutdown_event.wait()
        self.__server.stop(None)

    def EvaluateModel(self, request, context):
        if self.__not_serving_event.is_set():
            # Returns not available status if the servicer cannot receive new requests.
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return proto_factory.LearnerServiceProtoMessages\
                .construct_evaluate_model_response_pb()

        MetisLogger.info("Learner Servicer {} received model evaluation task.".format(
            self.learner.host_port_identifier()))
        self.__model_evaluation_requests += 1
        model_pb = request.model
        batch_size = request.batch_size
        metrics_pb = request.metrics
        evaluation_dataset_pb = request.evaluation_dataset
        # Blocking execution. Learner evaluates received model on its local datasets.
        # We do not stop any running evaluation tasks. Evaluation is critical! We need
        # all computed metrics to assess how well our models perform!
        model_evaluations_pb = self.learner.run_evaluation_task(
            model_pb, batch_size, evaluation_dataset_pb, metrics_pb,
            cancel_running_tasks=False,
            block=True,
            verbose=True)
        evaluate_model_response_pb = \
            proto_factory.LearnerServiceProtoMessages\
            .construct_evaluate_model_response_pb(model_evaluations_pb)
        return evaluate_model_response_pb

    def GetServicesHealthStatus(self, request, context):
        if self.__not_serving_event.is_set():
            # Returns not available status if the servicer cannot receive new requests.
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return proto_factory.ServiceCommonProtoMessages\
                .construct_get_services_health_status_request_pb()

        MetisLogger.info("Learner Servicer {} received a health status request.".format(
            self.learner.host_port_identifier()))
        services_status = {"server": self.__server is not None}
        return proto_factory\
            .ServiceCommonProtoMessages\
            .construct_get_services_health_status_response_pb(services_status=services_status)

    def RunTask(self, request, context):
        if self.__not_serving_event.is_set():
            # Returns not available status if the servicer cannot receive new requests.
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return proto_factory.LearnerServiceProtoMessages\
                .construct_run_task_response_pb()

        MetisLogger.info("Learner Servicer {} received local training task.".format(
            self.learner.host_port_identifier()))
        self.__community_models_received += 1
        federated_model = request.federated_model
        num_contributors = federated_model.num_contributors
        model_pb = federated_model.model
        learning_task_pb = request.task
        hyperparameters_pb = request.hyperparameters
        # Non-Blocking execution. Learner trains received task in the background and sends
        # the newly computed local model to the controller upon task completion. Upon receival
        # of a new training task the learner needs to cancel all running training tasks.
        is_task_submitted = self.learner.run_learning_task(
            learning_task_pb, hyperparameters_pb, model_pb,
            cancel_running_tasks=True,
            block=False,
            verbose=True)
        ack_pb = proto_factory.ServiceCommonProtoMessages.construct_ack_pb(
            status=is_task_submitted,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        run_task_response_pb = \
            proto_factory.LearnerServiceProtoMessages.construct_run_task_response_pb(ack_pb)
        return run_task_response_pb

    def ShutDown(self, request, context):
        MetisLogger.info("Learner Servicer {} received shutdown request.".format(
            self.learner.host_port_identifier()))
        # First, stop accepting new incoming requests.
        self.__not_serving_event.set()
        # Second, trigger a shutdown signal to learner's underlying execution engine
        # to release all resources and reply any pending tasks to the controller.
        # When shutdown is triggered we do not wait for any pending tasks.
        # We perform a forceful shutdown of all training and inference running tasks.
        # We only wait for the completion of the evaluation tasks.
        self.learner.shutdown(cancel_train_running_tasks=True,
                              cancel_eval_running_tasks=False,
                              cancel_infer_running_tasks=True)
        # Third, remove the learner from the federation.
        self.learner.leave_federation()
        # Fourth, issue servicer termination signal.
        # Reason we do this as the final step is because we want to the learner
        # to have completely shutdown its resources because if not it means it
        # is still alive and we cannot guarantee a proper shutdown.
        self.__shutdown_event.set()
        ack_pb = proto_factory.ServiceCommonProtoMessages.construct_ack_pb(
            status=True,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        shutdown_response_pb = \
            proto_factory.ServiceCommonProtoMessages.construct_shutdown_response_pb(ack_pb)
        return shutdown_response_pb
