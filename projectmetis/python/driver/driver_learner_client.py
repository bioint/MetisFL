import projectmetis.python.utils.proto_messages_factory as proto_factory

from projectmetis.python.utils.grpc_services import GRPCServerClient
from projectmetis.proto import learner_pb2_grpc


class DriverLearnerClient(GRPCServerClient):

    def __init__(self, learner_server_entity):
        super(DriverLearnerClient, self).__init__(learner_server_entity)

    def shutdown_learner(self):
        shutdown_request_pb = proto_factory\
            .ServiceCommonProtoMessages.construct_shutdown_request_pb()
        stub = learner_pb2_grpc.LearnerServiceStub(self._channel)
        response = stub.ShutDown(shutdown_request_pb)
        return response.ack.status
