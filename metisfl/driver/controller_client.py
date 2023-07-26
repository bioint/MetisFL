from metisfl.grpc.grpc_services import GRPCClient
from metisfl.proto import controller_pb2, service_common_pb2
from metisfl.utils.logger import MetisLogger
from metisfl.proto.proto_messages_factory import ModelProtoMessages


# FIXME(@stripeli) - logic here implies that requests go through without errors
# what about error handling?
class GRPCControllerClient(GRPCClient):
    def __init__(self,
                 controller_server_entity,
                 max_workers=1):
        super(GRPCControllerClient, self).__init__(
            controller_server_entity, max_workers)

    def check_health_status(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            get_services_health_status_request_pb = service_common_pb2.GetServicesHealthStatusRequest()
            MetisLogger.info("Requesting controller's health status.")
            response = self._stub.GetServicesHealthStatus(
                get_services_health_status_request_pb, timeout=_timeout)
            MetisLogger.info("Received controller's health status, {} - {}".format(
                self.grpc_endpoint.listening_endpoint, response))
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def get_community_model_evaluation_lineage(self, num_backtracks, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            request_pb = controller_pb2.GetCommunityModelEvaluationLineageRequest(
                num_backtracks=num_backtracks)
            MetisLogger.info(
                "Requesting community model evaluation lineage for {} backtracks.".format(num_backtracks))
            response = self._stub.GetCommunityModelEvaluationLineage(
                request_pb, timeout=_timeout)
            MetisLogger.info("Retrieved community model evaluation lineage.")
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def get_local_task_lineage(self, num_backtracks, learner_ids, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            request_pb = controller_pb2.GetLocalTaskLineageRequest(num_backtracks=num_backtracks,
                                                                   learner_ids=learner_ids)
            MetisLogger.info(
                "Requesting local model evaluation lineage for {} backtracks.".format(num_backtracks))
            response = self._stub.GetLocalTaskLineage(
                request_pb, timeout=_timeout)
            MetisLogger.info("Received local model evaluation lineage.")
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def get_participating_learners(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            request_pb = controller_pb2.GetParticipatingLearnersRequest()
            MetisLogger.info("Requesting number of participating learners.")
            response = self._stub.GetParticipatingLearners(
                request_pb, timeout=_timeout)
            MetisLogger.info("Received number of participating learners.")
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def get_runtime_metadata(self, num_backtracks, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            request_pb = controller_pb2.GetRuntimeMetadataLineageRequest(num_backtracks=num_backtracks)
            MetisLogger.info("Requesting runtime metadata lineage.")
            response = self._stub.GetRuntimeMetadataLineage(
                request_pb, timeout=_timeout)
            MetisLogger.info("Received runtime metadata lineage.")
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def replace_community_model(self, num_contributors, model_pb, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            federated_model_pb = ModelProtoMessages.construct_federated_model_pb(
                num_contributors, model_pb)
            request_pb = controller_pb2.ReplaceCommunityModelRequest(model=federated_model_pb)
            MetisLogger.info("Replacing controller's community model.")
            response = self._stub.ReplaceCommunityModel(
                request_pb, timeout=_timeout)
            MetisLogger.info("Replaced controller's community model.")
            return response.ack.status
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def shutdown_controller(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            request_pb = service_common_pb2.ShutDownRequest()
            MetisLogger.info("Sending shutdown request to controller {}.".format(
                self.grpc_endpoint.listening_endpoint))
            response = self._stub.ShutDown(request_pb, timeout=_timeout)
            MetisLogger.info("Sent shutdown request to controller {}.".format(
                self.grpc_endpoint.listening_endpoint))
            return response.ack.status
        return self.schedule_request(_request, request_retries, request_timeout, block)
