import os

import metisfl.learner.constants as constants
from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.grpc.grpc_controller_client import GRPCControllerClient
from metisfl.proto import metis_pb2


class FederationHelper:
    def __init__(
        self,
        learner_server_entity: metis_pb2.ServerEntity,
        controller_server_entity: metis_pb2.ServerEntity,
        learner_dataset: LearnerDataset,
        learner_credentials_fp: str,
    ) -> None:
        self.learner_server_entity = learner_server_entity
        self.learner_dataset = learner_dataset # FIXME: better not inject that here
        if not os.path.exists(learner_credentials_fp):
            os.mkdir(learner_credentials_fp)
            
        # TODO if we want to be more secure, we can dump an
        #  encrypted version of auth_token and learner_id
        self._learner_id_fp = os.path.join(learner_credentials_fp, constants.LEARNER_ID_FILE)
        self._auth_token_fp = os.path.join(learner_credentials_fp, constants.AUTH_TOKEN_FILE)
        self._learner_controller_client = GRPCControllerClient(controller_server_entity, max_workers=1)

    def host_port_identifier(self):
        return "{}:{}".format(
            self.learner_server_entity.hostname, self.learner_server_entity.port
     )

    def join_federation(self):
        # TODO If I create a learner controller instance once (without channel initialization)
        #  then the program hangs!
        dataset_metadata = self.learner_dataset.get_dataset_metadata()
        _, _, status = self._learner_controller_client.join_federation(
            self.learner_server_entity,
            self._learner_id_fp,
            self._auth_token_fp,
            dataset_metadata
        )
        return status        

    def mark_learning_task_completed(self, training_future):
        # If the returned future was completed successfully and was not cancelled,
        # meaning it did complete its running job, then notify the controller.
        if training_future.done() and not training_future.cancelled():
            completed_task_pb = training_future.result()
            self._learner_controller_client.mark_task_completed(
                completed_task_pb=completed_task_pb,
                block=False,
            )

    def leave_federation(self):
        status = self._learner_controller_client.leave_federation(block=False)
        # Make sure that all pending tasks have been processed.
        self._learner_controller_client.shutdown()
        return status
