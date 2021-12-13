import os
import signal

import projectmetis.python.utils.proto_messages_factory as proto_factory

from google.protobuf.timestamp_pb2 import Timestamp
from projectmetis.python.utils.grpc_services import GRPCServerMaxMsgLength
from projectmetis.proto import learner_pb2_grpc
from projectmetis.proto.metis_pb2 import ServerEntity
from projectmetis.python.learner.learner import Learner
from projectmetis.python.logging.metis_logger import MetisLogger


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(self, learner: Learner, learner_server_entity: ServerEntity,
                 servicer_workers=10, *args, **kwargs):
        self.learner = learner
        self.learner_server_entity = learner_server_entity
        self.servicer_workers = servicer_workers
        self.__community_models_received = 0
        self.__model_evaluation_requests = 0
        self.__server = None
        self.__pid = os.getpid()

    def init_servicer(self):
        self.__server = GRPCServerMaxMsgLength(max_workers=self.servicer_workers).server
        learner_pb2_grpc.add_LearnerServiceServicer_to_server(self, self.__server)
        self.__server.add_insecure_port(self.learner.host_port_identifier())
        self.__server.start()
        MetisLogger.info("Initialized Learner Servicer {}".format(
            self.learner.host_port_identifier()
        ))

    def wait_servicer(self):

        def handle_sigterm(*_):
            MetisLogger.info("Learner Servicer {} received shutdown signal.".format(
                self.learner.host_port_identifier()))
            # Shut down the server gracefully. Refuses new requests and waits for X seconds
            # for existing requests to complete. Returns immediately, but returns a threading.Event() object.
            all_rpcs_done_event = self.__server.stop(30)
            # Wait on the on threading.Event() to avoid premature exit.
            all_rpcs_done_event.wait(30)
            MetisLogger.info("Learner Servicer {} shutdown.".format(
                self.learner.host_port_identifier()))
        signal.signal(signal.SIGTERM, handle_sigterm)
        signal.signal(signal.SIGINT, handle_sigterm)

        if self.__server is not None:
            self.__server.wait_for_termination()
        else:
            raise RuntimeError("You need to first initialize LearnerServicer.")

    def EvaluateModel(self, request, context):
        self.__model_evaluation_requests += 1
        model_pb = request.model
        batch_size = request.batch_size
        metrics = request.metrics
        evaluation_dataset_pb = request.evaluation_dataset
        # Blocking execution. Learner evaluates received model on its local datasets.
        eval_result = self.learner.run_evaluation_task(
            model_pb, batch_size, evaluation_dataset_pb, metrics, verbose=True, block=True)
        model_evaluation_pb = \
            proto_factory.MetisProtoMessages.construct_model_evaluation_pb(eval_result)
        evaluate_model_response_pb = \
            proto_factory.LearnerServiceProtoMessages.construct_evaluate_model_response_pb(model_evaluation_pb)
        return evaluate_model_response_pb

    def GetServicesHealthStatus(self, request, context):
        pass

    def RunTask(self, request, context):
        self.__community_models_received += 1
        federated_model = request.federated_model
        num_contributors = federated_model.num_contributors
        model_pb = federated_model.model
        learning_task_pb = request.task
        hyperparameters_pb = request.hyperparameters

        # Non-Blocking execution. Learner trains received task in the background and sends
        # the newly computed local model to the controller upon task completion.
        is_task_submitted = self.learner.run_learning_task(
            learning_task_pb, hyperparameters_pb, model_pb, verbose=True, block=False)

        ack_pb = proto_factory.ServiceCommonProtoMessages.construct_ack_pb(
            status=is_task_submitted,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        run_task_response_pb = \
            proto_factory.LearnerServiceProtoMessages.construct_run_task_response_pb(ack_pb)
        return run_task_response_pb

    def ShutDown(self, request, context):
        # Issue termination signal
        os.kill(self.__pid, signal.SIGTERM)
        ack_pb = proto_factory.ServiceCommonProtoMessages.construct_ack_pb(
            status=True,
            google_timestamp=Timestamp().GetCurrentTime(),
            message=None)
        shutdown_response_pb = \
            proto_factory.ServiceCommonProtoMessages.construct_shutdown_response_pb(ack_pb)
        return shutdown_response_pb
