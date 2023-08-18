

import threading
from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from ..grpc.server import get_server
from ..proto import (learner_pb2, learner_pb2_grpc, model_pb2,
                     service_common_pb2)
from ..utils.fedenv import ServerParams
from .controller_client import GRPCControllerClient
from .learner import (Learner, try_call_evaluate, try_call_get_weights,
                      try_call_set_weights, try_call_train)
from .task_manager import TaskManager


class LearnerServer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(
        self,
        learner: Learner,
        client: GRPCControllerClient,
        task_manager: TaskManager,
        learner_params: ServerParams,
    ):
        """The Learner server. Impliments the LearnerServiceServicer endponits.

        Parameters
        ----------
        learner : Learner
            The Learner object. Must impliment the Learner interface.
        learner_params : ServerParams
            The server parameters of the Learner server.
        task_manager : TaskManager
            The task manager object. Udse to run tasks in a pool of workers.
        client : GRPCControllerClient
            The client object. Used to communicate with the controller.

        """
        self._learner = learner
        self._client = client
        self._task_manager = task_manager

        self._status = service_common_pb2.ServingStatus.UNKNOWN
        self._shutdown_event = threading.Event()

        self._server = get_server(
            server_params=learner_params,
            servicer=self,
            add_servicer_to_server_fn=learner_pb2_grpc.add_LearnerServiceServicer_to_server,
        )

    def start(self):
        """Starts the server."""
        self._server.start()
        self._status = service_common_pb2.ServingStatus.SERVING
        self._shutdown_event.wait()

    def GetHealthStatus(self) -> service_common_pb2.HealthStatusResponse:
        """Returns the health status of the server."""

        return service_common_pb2.HealthStatusResponse(
            ack=service_common_pb2.Ack(
                status=self._status == service_common_pb2.ServingStatus.SERVING,
            )
        )

    def GetModel(
        self,
        request: learner_pb2.GetModelRequest,
        context: Any
    ) -> learner_pb2.GetModelResponse:
        """Initializes the weights of the model.

        Parameters
        ----------
        request : learner_pb2.GetModelRequest
            An empty request. No parameters are needed.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        learner_pb2.GetModelResponse
            The response containing the model.

        """
        if not self._is_serving(context):
            return None

        model = try_call_get_weights(
            learner=self._learner,
        )

        return learner_pb2.GetModelResponse(
            model=model,
        )

    def SetInitialWeights(
        self,
        request: learner_pb2.SetInitialWeightsRequest,
        context: Any
    ) -> learner_pb2.SetInitialWeightsResponse:
        """Sets the initial weights of the model.

        Parameters
        ----------
        request : learner_pb2.SetInitialWeightsRequest
            The request containing the model.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        learner_pb2.SetInitialWeightsResponse
            The response containing the acknoledgement. The acknoledgement contains the status, i.e. True if the weights were set, False otherwise.
        """

        if not self._is_serving(context):
            return service_common_pb2.Ack(status=False)

        status = try_call_set_weights(
            learner=self._learner,
            model=request.model,
        )

        return learner_pb2.SetInitialWeightsResponse(
            ack=service_common_pb2.Ack(
                status=status,
                timestamp=Timestamp().GetCurrentTime(),
            )
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
        if not self._is_serving(context):
            return learner_pb2.EvaluateResponse(ack=None)

        metrics = try_call_evaluate(
            learner=self._learner,
            model=request.model,
            params=request.params,
        )

        # TODO: need to ensure that metrics is a dict containing the metrics in request.params.

        return learner_pb2.EvaluateResponse(
            metrics=metrics
        )

    def Train(
        self,
        request: learner_pb2.TrainRequest,
        context: Any
    ) -> learner_pb2.TrainResponse:
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
        learner_pb2.TrainResponse
            The response containing the acknoledgement. Always returns an acknoledgement with status True.  

        """
        if not self._is_serving(context):
            # TODO: Should we return an ack here? Check this.
            return learner_pb2.RunTaskResponse(ack=None)

        self._task_manager.run_task(
            task_fn=try_call_train,
            task_kwargs={
                'learner': self._learner,
                'model': request.model,
                'params': request.params,
            },
            callback=self._client.train_done,
        )

        ack = service_common_pb2.Ack(
            status=True,
            timestamp=Timestamp().GetCurrentTime(),
        )

        return learner_pb2.TrainResponse(
            ack=ack,
        )

    def ShutDown(self) -> learner_pb2.ShutDownResponse:
        """Shuts down the server."""

        self._status = service_common_pb2.ServingStatus.NOT_SERVING
        self._shutdown_event.set()

        return learner_pb2.ShutDownResponse(
            ack=service_common_pb2.Ack(
                status=True,
                timestamp=Timestamp().GetCurrentTime(),
            )
        )

    def _is_serving(self, context) -> bool:
        """Returns True if the server is serving, False otherwise."""

        if self._not_serving_event.is_set():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True
