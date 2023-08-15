

from typing import Any

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from ..grpc.server import get_server
from ..proto import (learner_pb2, learner_pb2_grpc, model_pb2,
                     service_common_pb2)
from ..utils.fedenv import ServerParams
from .learner import (Learner, try_call_evaluate, try_call_get_weights,
                      try_call_set_weights, try_call_train)
from .task_manager import TaskManager


class LearnerServer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(
        self,
        learner: Learner,
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
        """

        self._learner = learner
        self._task_manager = task_manager
        
        self._status = service_common_pb2.ServingStatus.UNKNOWN
        self._terminate = False
        
        self._server = get_server(
            server_params=learner_params,
            servicer=self,
            add_servicer_to_server_fn=learner_pb2_grpc.add_LearnerServiceServicer_to_server,
        )

    def start(self):
        """Starts the server."""
        self._server.start()
        self._status = service_common_pb2.ServingStatus.SERVING

    def GetHealthStatus(self) -> service_common_pb2.HealthStatusResponse:
        """Returns the health status of the server."""

        return service_common_pb2.HealthStatusResponse(status=self._status)

    def InitializeWeights(
        self,
        request: service_common_pb2.Empty,
        context: Any
    ) -> model_pb2.Model:
        """Initializes the weights of the model.

        Parameters
        ----------
        request : service_common_pb2.Empty
            An empty request. No parameters are needed to initialize the weights.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        model_pb2.Model
            The initialized model weights.
        """
        if not self._is_serving(context):
            return None

        # TODO:

        return model

    def SetInitialWeights(
        self,
        request: model_pb2.Model,
        context: Any
    ) -> service_common_pb2.Ack:
        """Sets the initial weights of the model.

        Parameters
        ----------
        request : model_pb2.Model
            The ProtoBuf model containing the initial weights.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The acknoledgement contain the status, i.e. True if the weights were set successfully, False otherwise.
        """
        
        if not self._is_serving(context):
            return service_common_pb2.Ack(status=False)
        
        status = try_call_set_weights(
            learner=self._learner,
            model=request,
        )
        
        return service_common_pb2.Ack(
            status=status,
            timestamp=Timestamp().GetCurrentTime(),
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

        return learner_pb2.EvaluateResponse(
            metrics=metrics
        )

    def Train(
        self,
        request: learner_pb2.TrainRequest,
        context: Any
    ) -> service_common_pb2.Ack:
        """Training endpoint.

        Parameters
        ----------
        request : learner_pb2.TrainRequest
            The request containing the model and training parameters.
        context : Any
            The gRPC context of the request.

        Returns
        -------
        service_common_pb2.Ack
            The acknoledgement contain the status, i.e. True if the training has started, False otherwise.
            
        """
        if not self._is_serving(context):
            # TODO: Should we return an ack here? Check this.
            return learner_pb2.RunTaskResponse(ack=None)

        status = try_call_train(
            learner=self._learner,
            model=request.model,
            params=request.params,
        )

        return service_common_pb2.Ack(
            status=status,
            timestamp=Timestamp().GetCurrentTime(),
        )

    def ShutDown(self) -> service_common_pb2.Ack:
        """Shuts down the server."""

        self._status = service_common_pb2.ServingStatus.NOT_SERVING
        self._terminate = True

        return service_common_pb2.Ack(
            status=True,
            timestamp=Timestamp().GetCurrentTime(),
        )

    def _is_serving(self, context):
        """Returns True if the server is serving, False otherwise."""

        if self._not_serving_event.is_set():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True
