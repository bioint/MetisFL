import grpc
import time

import src.python.utils.proto_messages_factory as proto_factory

from src.python.utils.metis_logger import MetisLogger
from src.python.utils.grpc_services import GRPCServerClient
from src.proto import learner_pb2_grpc


class GRPCLearnerClient(GRPCServerClient):

    def __init__(self, learner_server_entity, max_workers=1):
        super(GRPCLearnerClient, self).__init__(learner_server_entity, max_workers)
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

    def check_health_status(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            get_services_health_status_request_pb = \
                proto_factory.ServiceCommonProtoMessages. \
                construct_get_services_health_status_request_pb()
            MetisLogger.info("Checking health status of learner {}.".format(self._server_entity))
            health_status = self._stub.GetServicesHealthStatus(get_services_health_status_request_pb, timeout=_timeout)
            MetisLogger.info("Health status of learner {} - {}".format(self._server_entity, health_status))
            return health_status

        if request_retries > 1:
            future = self.executor.schedule(function=self.request_with_timeout,
                                            args=(_request, request_timeout, request_retries))
        else:
            future = self.executor.schedule(_request)

        if block:
            return future.result()
        else:
            self.executor_pool.put(future)

    def shutdown_learner(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            shutdown_request_pb = proto_factory \
                .ServiceCommonProtoMessages.construct_shutdown_request_pb()
            MetisLogger.info("Sending shutdown request to learner {}.".format(self._server_entity))
            response = self._stub.ShutDown(shutdown_request_pb, timeout=_timeout)
            MetisLogger.info("Sent shutdown request to learner {}, response: {}.".format(self._server_entity, response))
            return response.ack.status

        if request_retries > 1:
            future = self.executor.schedule(function=self.request_with_timeout,
                                            args=(_request, request_timeout, request_retries))
        else:
            future = self.executor.schedule(_request)

        if block:
            return future.result()
        else:
            self.executor_pool.put(future)
