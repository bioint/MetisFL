import grpc

from metisfl import config
from metisfl.grpc.grpc_services import GRPCClient
from metisfl.proto import controller_pb2, controller_pb2_grpc, metis_pb2
from metisfl.utils.metis_logger import MetisLogger
from metisfl.proto.proto_messages_factory import MetisProtoMessages


class GRPCControllerClient(GRPCClient):
    # When issuing the join federation request, the learner/client needs also to  share
    # the public certificate of its servicer with the controller, in order  to receive
    # new incoming requests. The certificate needs to be in bytes (stream) format.
    # Most importantly, the learner/client must not share  its private key with the
    # controller. Therefore, we need to clear the private key to make sure it is not
    # released to the controller. To achieve this, we generate a new ssl configuration
    # only with the value of the public certificate.
    # If it is the first time that the learner joins the federation, then both
    # learner id and authentication token are saved on disk to appropriate files.
    # If the learner has previously joined the federation, then an error
    # grpc.StatusCode.ALREADY_EXISTS is raised and the existing/already saved
    # learner id and authentication token are read/loaded from the disk.

    def __init__(self,
                 controller_server_entity,
                 learner_server_entity,
                 dataset_metadata,
                 max_workers=1):
        super(GRPCControllerClient, self).__init__(
            controller_server_entity, max_workers)
        self._learner_id_fp = config.get_learner_id_fp(
            learner_server_entity.port)
        self._auth_token_fp = config.get_auth_token_fp(
            learner_server_entity.port)
        self._learner_server_entity = learner_server_entity
        self._dataset_metadata = dataset_metadata
        self._stub = controller_pb2_grpc.ControllerServiceStub(self._channel)

        # These must be set after joining the federation, provided by the controller
        self._learner_id = None
        self._auth_token = None

    def join_federation(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            join_federation_request_pb = _get_join_request_pb(self._learner_server_entity,
                                                              self._dataset_metadata)
            self._join_federation(join_federation_request_pb, timeout=_timeout)
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def _join_federation(self, join_federation_request_pb, timeout=None):
        try:
            MetisLogger.info("Joining federation, learner {}.".format(
                self.grpc_endpoint.listening_endpoint))
            response = self._stub.JoinFederation(
                join_federation_request_pb, timeout=timeout)
            learner_id, auth_token, status = \
                response.learner_id, response.auth_token, response.ack.status
            # override file contents or create file if not exists
            # FIXME: need to handle file open/write exceptions
            open(self._learner_id_fp, "w+").write(learner_id.strip())
            open(self._auth_token_fp, "w+").write(auth_token.strip())
            MetisLogger.info(
                "Joined federation with assigned id: {}".format(learner_id))
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.ALREADY_EXISTS:
                learner_id = open(self._learner_id_fp, "r").read().strip()
                auth_token = open(self._auth_token_fp, "r").read().strip()
                status = True
                MetisLogger.info(
                    "Learner re-joined federation with assigned id: {}".format(learner_id))
            else:
                raise RuntimeError(
                    "Unhandled grpc error: {}".format(rpc_error))
        self._learner_id = learner_id
        self._auth_token = auth_token
        return learner_id, auth_token, status

    def leave_federation(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            leave_federation_request_pb = controller_pb2.LeaveFederationRequest(learner_id=self._learner_id,
                                                                                auth_token=self._auth_token)
            MetisLogger.info(
                "Leaving federation, learner {}.".format(self._learner_id))
            response = self._stub.LeaveFederation(
                leave_federation_request_pb, timeout=_timeout)
            MetisLogger.info(
                "Left federation, learner {}.".format(self._learner_id))
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def mark_task_completed(self, completed_task_pb, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            mark_task_completed_request_pb = controller_pb2.MarkTaskCompletedRequest(learner_id=self._learner_id,
                                                                                     auth_token=self._auth_token,
                                                                                     task=completed_task_pb)
            MetisLogger.info(
                "Sending local completed task, learner {}.".format(self._learner_id))
            response = self._stub.MarkTaskCompleted(
                mark_task_completed_request_pb, timeout=_timeout)
            MetisLogger.info(
                "Sent local completed task, learner {}.".format(self._learner_id))
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)


def _get_join_request_pb(learner_server_entity, dataset_metadata):
    # @stripeli why is the server enntity being recreated here?
    learner_server_entity_public = metis_pb2.ServerEntity(
        hostname=learner_server_entity.hostname,
        port=learner_server_entity.port,
        public_certificate_file=learner_server_entity.public_certificate_file,
        private_key_file=None,
    )
    dataset_spec_pb = MetisProtoMessages.construct_dataset_spec_pb(
        num_training_examples=dataset_metadata["train_dataset_size"],
        num_validation_examples=dataset_metadata["validation_dataset_size"],
        num_test_examples=dataset_metadata["test_dataset_size"],
        training_spec=dataset_metadata["train_dataset_specs"],
        validation_spec=dataset_metadata["validation_dataset_specs"],
        test_spec=dataset_metadata["test_dataset_specs"],
        is_classification=dataset_metadata["is_classification"],
        is_regression=dataset_metadata["is_regression"]
    )
    join_federation_request_pb = controller_pb2.JoinFederationRequest(server_entity=learner_server_entity_public,
                                                                      local_dataset_spec=dataset_spec_pb)
    return join_federation_request_pb
