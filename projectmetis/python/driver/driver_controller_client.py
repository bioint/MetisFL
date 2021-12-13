import projectmetis.python.utils.proto_messages_factory as proto_factory

from projectmetis.python.utils.grpc_services import GRPCServerClient
from projectmetis.proto import controller_pb2_grpc


class DriverControllerClient(GRPCServerClient):

    def __init__(self, controller_server_entity):
        super(DriverControllerClient, self).__init__(controller_server_entity)

    def shutdown_controller(self):
        shutdown_request_pb = proto_factory \
            .ServiceCommonProtoMessages.construct_shutdown_request_pb()
        stub = controller_pb2_grpc.ControllerServiceStub(self._channel)
        response = stub.ShutDown(shutdown_request_pb)
        return response.ack.status
