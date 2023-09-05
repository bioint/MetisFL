

from typing import Dict, List, Optional
from metisfl.common.client import get_client
from metisfl.common.types import ClientParams

from metisfl.proto import (controller_pb2, controller_pb2_grpc, learner_pb2, learner_pb2_grpc,
                           model_pb2, service_common_pb2)


def get_learner_id(
    hostname: str,
    port: int,
) -> str:
    """Gets the learner id from the hostname and port."""
    return f"{hostname}:{port}"


class LearnerManager:

    # learner_id -> {}
    learners: Dict[str, controller_pb2.Learner] = {}
    client_params: Dict[str, ClientParams] = {}
    train_params: learner_pb2.TrainParams = None
    eval_params: learner_pb2.EvalParams = None

    # task_id -> {}
    tasks: Dict[str, learner_pb2.Task] = {}
    train_results: Dict[str, controller_pb2.TrainResult] = {}
    eval_results: Dict[str, learner_pb2.EvalResult] = {}

    # learner_id -> {}
    last_train_results: Dict[str, controller_pb2.TrainResult] = {}
    num_training_examples: Dict[str, int] = {}

    def __init__(self, learner):
        self.learner = learner

    def add_learner(self, learner: controller_pb2.Learner) -> str:
        """Adds a learner to the controller.

        Parameters
        ----------
        learner : controller_pb2.Learner
            The learner to be added.
        """
        learner_id = get_learner_id(
            hostname=learner.hostname,
            port=learner.port,
        )

        if learner_id in self.learners:
            raise ValueError(f"Learner {learner_id} already exists.")

        self.learners[learner_id] = learner
        self.client_params[learner_id] = ClientParams(
            hostname=learner.hostname,
            port=learner.port,
            root_certificate=learner.root_certificate_bytes,
        )

        return learner_id

    def remove_learner(self, learner_id: str) -> None:
        """Removes a learner from the controller.

        Parameters
        ----------
        learner_id : str
            The learner id.
        """
        if learner_id not in self.learners:
            raise ValueError(f"Learner {learner_id} does not exist.")

        self.learners.pop(learner_id)
        self.clients.pop(learner_id)

    def get_client(self, learner_id: str):
        return get_client(
            stub_class=controller_pb2_grpc.ControllerServiceStub,
            client_params=self.client_params[learner_id],
        )

    def schedule_train(
        self,
        task: learner_pb2.Task,
        model: model_pb2.Model,
        train_params: learner_pb2.TrainParams,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = False
    ) -> service_common_pb2.Ack:

        with self.get_client() as client:

            stub: learner_pb2_grpc.LearnerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):

                request = controller_pb2.TrainRequest(
                    task=task,
                    model=model,
                    params=train_params
                )

                return stub.Train(request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def schedule_evaluate(
        self,
        task: learner_pb2.Task,
        model: model_pb2.Model,
        eval_params: learner_pb2.EvaluationParams,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = False
    ) -> service_common_pb2.Ack:

        with self.get_client() as client:

            stub: learner_pb2_grpc.LearnerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):

                request = controller_pb2.EvaluateRequest(
                    task=task,
                    model=model,
                    params=eval_params
                )

                return stub.Evaluate(request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self):
        """Shuts down the client."""
        with self.get_client() as client:
            client[2].shutdown()

    def get_num_training_examples(self, learner_ids: List[str]) -> Dict[str, int]:
        """Gets the number of training examples for the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, int]
            The number of training examples for each learner.
        """
        num_training_examples = {}
        for learner_id in learner_ids:
            num_training_examples[learner_id] = self.num_training_examples.get(
                learner_id, 0
            )

    def get_num_completed_batches(self, learner_ids: List[str]) -> Dict[str, int]:
        """Gets the number of completed batches for the learners.

        Parameters
        ----------
        learner_ids : List[str]
            The learner ids.

        Returns
        -------
        Dict[str, int]
            The number of completed batches for each learner.
        """
        num_completed_batches = {}
        for learner_id in learner_ids:
            num_completed_batches[learner_id] = self.last_train_results.get(
                learner_id, 0
            )  # FIXME: this is not correct

    def get_learner_ids(self) -> List[str]:
        """Gets the learner ids.

        Returns
        -------
        List[str]
            The learner ids.
        """
        return list(self.learners.keys())

    def get_learner_id(self, task_id: str) -> str:
        """Gets the learner id of a task."""
        return self.tasks[task_id].learner_id

    def update_train_result(
        self,
        task: learner_pb2.Task,
        learner_id: str,
        train_results: controller_pb2.TrainResults,
    ) -> None:
        """Updates the train result of a learner.

        Parameters
        ----------
        task : learner_pb2.Task
            The newly completed task.
        learner_id : str
            The learner id.
        train_result : controller_pb2.TrainResults
            The train result of the task.
        """
        task_id = task.id
        self.train_results[task_id] = train_results
        self.last_train_results[learner_id] = train_results
        self.tasks[task_id].received_at = task.received_at
        self.tasks[task_id].completed_at = task.completed_at
