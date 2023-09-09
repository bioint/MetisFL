

import threading
from typing import Any, Tuple

import grpc
from google.protobuf.json_format import MessageToDict
from loguru import logger

from metisfl.common.formatting import get_timestamp
from metisfl.common.server import get_server
from metisfl.common.types import ServerParams
from metisfl.learner.client import GRPCClient
from metisfl.learner.learner import (Learner, try_call_evaluate,
                                     try_call_get_weights,
                                     try_call_set_weights, try_call_train)
from metisfl.learner.message import MessageHelper
from metisfl.learner.tasks import TaskManager
from metisfl.proto import (learner_pb2, learner_pb2_grpc, model_pb2,
                           service_common_pb2)


class LearnerServer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(
        self,
        learner: Learner,
        client: GRPCClient,
        message_helper: MessageHelper,
        task_manager: TaskManager,
        server_params: ServerParams,
    ):
        """The Learner server. Impliments the LearnerServiceServicer endponits.

        Parameters
        ----------
        learner : Learner
            The Learner object. Must impliment the Learner interface.
        client : GRPCControllerClient
            The client object. Used to communicate with the controller.
        message_helper : MessageHelper
            The message helper object. Used to convert ProtoBuf objects
        task_manager : TaskManager
            The task manager object. Udse to run tasks in a pool of workers.
        server_params : ServerParams
            The server parameters of the Learner server.

        """
        self.learner = learner
        self.client = client
        self.message_helper = message_helper
        self.task_manager = task_manager

        self.status = service_common_pb2.ServingStatus.UNKNOWN
        self.shutdown_event = threading.Event()
        self.server_params = server_params

        self.server = get_server(
            server_params=server_params,
            servicer=self,
            add_servicer_to_server_fn=learner_pb2_grpc.add_LearnerServiceServicer_to_server,
        )

    def start(self):
        """Starts the server. This is a blocking call and will block until the server is shutdown."""

        self.server.start()  # TODO: anyway this could fail?
        self.status = service_common_pb2.ServingStatus.SERVING
        logger.success("Learner server started. Listening on: {}:{} with SSL: {}".format(
            self.server_params.hostname,
            self.server_params.port,
            "ENABLED" if self.is_ssl() else "DISABLED",
        ))
        self.shutdown_event.wait()

    def GetHealthStatus(self) -> service_common_pb2.HealthStatusResponse:
        """Returns the health status of the server."""

        return service_common_pb2.HealthStatusResponse(
            status=self.status,
        )

    def GetModel(
        self,
        _: service_common_pb2.Empty,
        context: Any
    ) -> model_pb2.Model:
        """Initializes the weights of the model.

        Parameters
        ----------
        _ : service_common_pb2.Empty
            An empty request. No parameters are needed.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        model_pb2.Model
            The ProtoBuf object containing the model.

        """
        if not self.is_serving(context):
            return None

        weights = try_call_get_weights(
            learner=self.learner,
        )

        return self.message_helper.weights_to_model_proto(weights)

    def SetInitialModel(
        self,
        model: model_pb2.Model,
        context: Any
    ) -> service_common_pb2.Ack:
        """Sets the initial weights of the model.

        Parameters
        ----------
        request : model_pb2.Model
            The ProtoBuf object containing the model.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement.
        """

        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        weights = self.message_helper.model_proto_to_weights(model)

        try_call_set_weights(
            learner=self.learner,
            weights=weights,
        )

        logger.success("Initial model set.")

        return service_common_pb2.Ack(
            status=True,
            timestamp=get_timestamp(),
        )

    def Evaluate(
        self,
        request: learner_pb2.EvaluateRequest,
        context: Any
    ) -> learner_pb2.EvaluateResponse:
        """Evaluation endpoint. Evaluates the given model.

        Parameters
        ----------
        request : learner_pb2.EvaluateRequest
            The request containing the model and evaluation parameters.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        learner_pb2.EvaluateResponse
            The response containing the evaluation metrics.
        """
        if not self.is_serving(context):
            return learner_pb2.EvaluateResponse(ack=None)

        logger.info("Received evaluation task with id: {}".format(
            request.task.id))

        weights = self.message_helper.model_proto_to_weights(request.model)
        params = MessageToDict(request.params)

        received_at = get_timestamp()
        metrics = try_call_evaluate(
            learner=self.learner,
            weights=weights,
            params=params,
        )

        logger.success("Evaluation task with id: {} completed.".format(
            request.task.id))

        return learner_pb2.EvaluateResponse(
            task=learner_pb2.Task(
                id=request.task.id,
                sent_at=request.task.sent_at,
                received_at=received_at,
                completed_at=get_timestamp(),
            ),
            results=learner_pb2.EvaluationResults(
                metrics=metrics,
            ),
        )

    def Train(
        self,
        request: learner_pb2.TrainRequest,
        context: Any
    ) -> service_common_pb2.Ack:
        """Training endpoint. Training happens asynchronously in a seperate process. 
            The Learner server responds with an acknoledgement after receiving the request.
            When training is done, the client calls the TrainDone Controller endpoint.

        Parameters
        ----------
        request : learner_pb2.TrainRequest
            The request containing the model and training parameters.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The response containing the acknoledgement with the Status set to True.

        """
        if not self.is_serving(context):
            return service_common_pb2.Ack(status=False)

        logger.info("Received training task with id: {}".format(
            request.task.id))

        task: learner_pb2.Task = request.task
        weights = self.message_helper.model_proto_to_weights(request.model)
        params = MessageToDict(request.params)

        new_task = learner_pb2.Task(
            id=task.id,
            sent_at=task.sent_at,
            received_at=get_timestamp(),
        )

        def train_out_to_callback_fn(train_out: Tuple[Any]) -> Tuple[Any]:
            return (
                new_task,  # task
                train_out[0],  # weights
                train_out[1],  # metrics
                train_out[2]  # metadata
            )

        self.task_manager.run_task(
            task_fn=try_call_train,
            task_kwargs={
                'learner': self.learner,
                'weights': weights,
                'params': params,
            },
            callback=self.client.train_done,
            task_out_to_callback_fn=train_out_to_callback_fn,
        )

        return service_common_pb2.Ack(
            status=True,
            timestamp=get_timestamp(),
        )

    def ShutDown(
        self,
        _: service_common_pb2.Empty,
        __: Any
    ) -> service_common_pb2.Ack:
        """Shuts down the server."""

        self.status = service_common_pb2.ServingStatus.NOT_SERVING
        self.shutdown_event.set()

        return service_common_pb2.Ack(
            status=True,
            timestamp=get_timestamp(),
        )

    def is_serving(self, context) -> bool:
        """Returns True if the server is serving, False otherwise."""

        if self.status != service_common_pb2.ServingStatus.SERVING:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True

    def is_ssl(self) -> bool:
        """Returns True if the server is using SSL, False otherwise."""

        return self.server_params.root_certificate is not None
