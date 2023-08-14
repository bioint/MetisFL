

import config
import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from ..proto import (learner_pb2, learner_pb2_grpc, model_pb2,
                     service_common_pb2)
from .learner import Learner


class LearnerServicer(learner_pb2_grpc.LearnerServiceServicer):

    def __init__(
        self,
        learner: Learner
    ):
        """Initializes the LearnerServicer."""

        self._learner = learner
        self._status = service_common_pb2.ServingStatus.UNKNOWN

    def start(self):
        pass

    def GetHealthStatus(self) -> service_common_pb2.HealthStatusResponse:
        return service_common_pb2.HealthStatusResponse(status=self._status)

    def InitializeWeights(self, request, context) -> model_pb2.Model:
        pass

    def SetInitialWeights(self, request, context) -> service_common_pb2.Ack:
        pass

    def Evaluate(self, request, context) -> learner_pb2.EvaluateResponse:
        pass 

    def Train(self, request, context) -> service_common_pb2.Ack:
        if self._is_serving(context) is False:
            return learner_pb2.RunTaskResponse(ack=None)

        is_task_submitted = self._learner_executor.run_learning_task(
            callback=self._client.mark_task_completed,
            learning_task_pb=request.task,
            model_pb=request.federated_model.model,
            hyperparameters_pb=request.hyperparameters,
            cancel_running_tasks=True,
            block=False,
            verbose=True)

        return service_common_pb2.Ack(
            status=is_task_submitted,
            timestamp=Timestamp().GetCurrentTime(),
        )

    def ShutDown(self) -> service_common_pb2.Ack:
        self._status = service_common_pb2.ServingStatus.NOT_SERVING
        
        self._learner_executor.shutdown(
            cancel_running=config.CANCEL_RUNNING_ON_SHUTDOWN)

        return service_common_pb2.Ack(
            status=True,
            timestamp=Timestamp().GetCurrentTime(),
        )

    def _is_serving(self, context):
        if self._not_serving_event.is_set():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True
