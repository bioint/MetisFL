from metisfl.grpc import GRPCServerClient
from metisfl.proto import learner_pb2_grpc
from metisfl.utils import ServiceCommonProtoMessages
from metisfl.utils.metis_logger import MetisLogger


class GRPCLearnerClient(GRPCServerClient):

    def __init__(self, learner_server_entity, max_workers=1):
        super(GRPCLearnerClient, self).__init__(learner_server_entity, max_workers)
        self._stub = learner_pb2_grpc.LearnerServiceStub(self._channel)

    def shutdown_learner(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            shutdown_request_pb = ServiceCommonProtoMessages.construct_shutdown_request_pb()
            MetisLogger.info("Sending shutdown request to learner {}.".format(
                self.grpc_endpoint.listening_endpoint))
            response = self._stub.ShutDown(shutdown_request_pb, timeout=_timeout)
            MetisLogger.info("Sent shutdown request to learner {}, response: {}.".format(
                self.grpc_endpoint.listening_endpoint, response))
            return response.ack.status
        self.schedule_request(_request, request_retries, request_timeout, block)
