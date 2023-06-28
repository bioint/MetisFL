from metisfl.utils.metis_logger import MetisLogger
from metisfl.grpc.grpc_services import GRPCServerClient
from metisfl.proto import learner_pb2_grpc
from metisfl.utils.proto_messages_factory import ServiceCommonProtoMessages


class GRPCLearnerClient(GRPCServerClient):

    def __init__(self, learner_server_entity, max_workers=1):
        super(GRPCLearnerClient, self).__init__(learner_server_entity, max_workers)
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

    def check_health_status(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            get_services_health_status_request_pb = ServiceCommonProtoMessages. \
                    construct_get_services_health_status_request_pb()
            MetisLogger.info("Checking health status of learner {}.".format(
                self.grpc_endpoint.listening_endpoint))
            health_status = self._stub.GetServicesHealthStatus(
                get_services_health_status_request_pb, timeout=_timeout)
            MetisLogger.info("Health status of learner {} - {}".format(
                self.grpc_endpoint.listening_endpoint, health_status))
            return health_status
        return self._schedule_request(_request, request_retries, request_timeout, block)

    def shutdown_learner(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            shutdown_request_pb = ServiceCommonProtoMessages.construct_shutdown_request_pb()
            MetisLogger.info("Sending shutdown request to learner {}.".format(
                self.grpc_endpoint.listening_endpoint))
            response = self._stub.ShutDown(shutdown_request_pb, timeout=_timeout)
            MetisLogger.info("Sent shutdown request to learner {}, response: {}.".format(
                self.grpc_endpoint.listening_endpoint, response))
            return response.ack.status
        self._schedule_request(_request, request_retries, request_timeout, block)

    def _schedule_request(self, request, request_retries=1, request_timeout=None, block=True):
        if request_retries > 1:
            future = self.executor.schedule(function=self.request_with_timeout,
                                            args=(request, request_timeout, request_retries)) # FIXME: shouldn't we pass request_retries - 1?
        else:
            future = self.executor.schedule(request)

        if block:
            return future.result()
        else:
            self.executor_pool.put(future)
