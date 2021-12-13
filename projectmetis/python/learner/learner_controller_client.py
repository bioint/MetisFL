import grpc

import projectmetis.python.utils.proto_messages_factory as proto_factory
import projectmetis.python.models.model_dataset as model_dataset

from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.utils.grpc_services import GRPCServerClient
from projectmetis.proto import controller_pb2, controller_pb2_grpc


class LearnerControllerClient(GRPCServerClient):

    def __init__(self, controller_server_entity):
        super(LearnerControllerClient, self).__init__(controller_server_entity)

    def join_federation(self, learner_server_entity,
                        learner_id_fp, auth_token_fp,
                        train_dataset, test_dataset,
                        validation_dataset) \
            -> controller_pb2.JoinFederationResponse:

        dataset_spec_pb = proto_factory.MetisProtoMessages.construct_dataset_spec_pb(
            num_training_examples=train_dataset.get_size(),
            num_validation_examples=validation_dataset.get_size(),
            num_test_examples=test_dataset.get_size(),
            training_spec=train_dataset.get_model_dataset_specifications(),
            validation_spec=validation_dataset.get_model_dataset_specifications(),
            test_spec=test_dataset.get_model_dataset_specifications(),
            is_classification=isinstance(train_dataset, model_dataset.ModelDatasetClassification),
            is_regression=isinstance(train_dataset, model_dataset.ModelDatasetRegression))

        join_federation_request_pb = proto_factory.ControllerServiceProtoMessages \
            .construct_join_federation_request_pb(server_entity_pb=learner_server_entity,
                                                  local_dataset_spec_pb=dataset_spec_pb)
        stub = controller_pb2_grpc.ControllerServiceStub(self._channel)

        # If it is the first time that the learner joins the federation, then both
        # learner id and authentication token are saved on disk to appropriate files.
        # If the learner has previously joined the federation, then an error
        # grpc.StatusCode.ALREADY_EXISTS is raised and the existing/already saved
        # learner id and authentication token are read/loaded from the disk.
        try:
            response = stub.JoinFederation(join_federation_request_pb)
            learner_id, auth_token, status = \
                response.learner_id, response.auth_token, response.ack.status
            # override file contents or create file if not exists
            open(learner_id_fp, "w+").write(learner_id.strip())
            open(auth_token_fp, "w+").write(auth_token.strip())
            MetisLogger.info("Learner joined federation with assigned id: {}".format(learner_id))
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.ALREADY_EXISTS:
                learner_id = open(learner_id_fp, "r").read().strip()
                auth_token = open(auth_token_fp, "r").read().strip()
                status = True
                MetisLogger.info("Learner re-joined federation with assigned id: {}".format(learner_id))
            else:
                raise RuntimeError("Unhandled grpc error: {}".format(rpc_error))
        return learner_id, auth_token, status

    def leave_federation(self, learner_id, auth_token) -> controller_pb2.LeaveFederationResponse:
        leave_federation_request_pb = proto_factory.ControllerServiceProtoMessages \
            .construct_leave_federation_request_pb(learner_id=learner_id, auth_token=auth_token)
        stub = controller_pb2_grpc.ControllerServiceStub(self._channel)
        response = stub.LeaveFederation(leave_federation_request_pb)
        return response

    def mark_task_completed(self, learner_id, auth_token, model_weights, model_meta, aux_metadata=""):
        model_vars = []
        for widx, weight in enumerate(model_weights):
            tensor_pb = proto_factory.ModelProtoMessages.construct_tensor_pb_from_nparray(weight)
            model_var = proto_factory.ModelProtoMessages.construct_model_variable_pb(name="arr_{}".format(widx),
                                                                                     trainable=True,
                                                                                     tensor_pb=tensor_pb)
            model_vars.append(model_var)
        model_pb = proto_factory.ModelProtoMessages.construct_model_pb(model_vars)
        task_execution_meta_pb = model_meta.get_task_execution_metadata_pb()
        completed_learning_task_pb = proto_factory.MetisProtoMessages\
            .construct_completed_learning_task_pb(model_pb=model_pb,
                                                  task_execution_metadata_pb=task_execution_meta_pb,
                                                  aux_metadata=aux_metadata)
        mark_task_completed_request_pb = proto_factory.ControllerServiceProtoMessages\
            .construct_mark_task_completed_request_pb(learner_id=learner_id,
                                                      auth_token=auth_token,
                                                      completed_learning_task_pb=completed_learning_task_pb)
        stub = controller_pb2_grpc.ControllerServiceStub(self._channel)
        response = stub.MarkTaskCompleted(mark_task_completed_request_pb)
        return response
