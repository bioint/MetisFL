from typing import Any, Dict, List, Optional

import grpc
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

from metisfl.proto import learner_pb2

from metisfl.common.client import get_client
from metisfl.common.logger import MetisLogger
from metisfl.common.types import ClientParams, ServerParams
from metisflproto import controller_pb2, controller_pb2_grpc, service_common_pb2
from metisfl.message_helper import MessageHelper


def read_certificate(fp: str) -> bytes:
    if fp is None:
        return None
    with open(fp, "rb") as f:
        return f.read()


class GRPCClient(object):

    def __init__(
        self,
        client_params: ClientParams,
        message_helper: MessageHelper,
        learner_id_fp: str,
        max_workers: Optional[int] = 1
    ):
        """A gRPC client used from the Learner to communicate with the Controller.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters. Used by the learner to connect to the Controller.
        message_helper : MessageHelper
            The MessageHelper object used to serialize the requests.
        learner_id_fp : str
            The file where the learner id is stored. 
        max_workers : Optional[int], (default: 1)
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._client_params = client_params
        self._message_helper = message_helper
        self._learner_id_fp = learner_id_fp
        self._max_workers = max_workers

        # Must be initialized after joining the federation
        self._learner_id = None
        self._auth_token = None

    def _get_client(self):
        return get_client(
            stub_class=controller_pb2_grpc.ControllerServiceStub,
            client_params=self._client_params,
            max_workers=self._max_workers
        )

    def join_federation(
        self,
        num_training_examples: int,
        server_params: ServerParams,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.Ack:
        """Sends a request to the controller to join the federation.

        Parameters
        ----------
        num_training_examples : int
            The number of training examples of the local dataset.
        server_params : ServerParams
            The server parameters of the Learner server. They are sent to the Controller 
            when joining the federation so that the Controller can connect to the Learner Server.
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack  
            The response Proto object with the Ack.
        """
        with self._get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):

                request = controller_pb2.Learner(
                    hostname=server_params.hostname,
                    port=server_params.port,
                    root_certificate_bytes=read_certificate(
                        server_params.root_certificate),
                    public_certificate_bytes=read_certificate(
                        server_params.server_certificate),
                    num_training_examples=num_training_examples
                )

                return self._join_federation(stub, request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def leave_federation(
        self,
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.Ack:
        """Sends a request to the Controller to leave the federation.

        Parameters
        ----------
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : _type_, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        Raises
        ------
        RuntimeError
            If the learner id does not exist, 
            which means that the Learner has not joined the federation.
        """

        if not self._has_learner_id():
            MetisLogger.warning(
                "Cannot leave federation before joining the federation.")
            return

        with self._get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):
                request = controller_pb2.LearnerId(
                    id=self._learner_id,
                )
                return stub.LeaveFederation(
                    request=request,
                    timeout=_timeout
                )

            return schedule(_request, request_retries, request_timeout, block)

    def train_done(
        self,
        task: learner_pb2.Task,
        weights: List[np.ndarray],
        metrics: Dict[str, Any],
        metadata: Dict[str, str],
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.Ack:
        """Sends the completed task to the Controller.

        Parameters
        ----------
        task : learner_pb2.Task
            The task Proto object. 
        weights : List[np.ndarray]
            The weights of the model.
        metrics : Dict[str, Any]
            The metrics produced during training.
        metadata : Dict[str, str]
            The metadata to be sent.
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        Raises
        ------
        RuntimeError
            If the learner id does not exist,
            which means that the Learner has not joined the federation.
        """

        if not self._has_learner_id():
            raise RuntimeError(
                "Cannot send train done before joining the federation.")

        with self._get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            task = learner_pb2.Task(
                id=task.id,
                received_at=task.received_at,
                completed_at=Timestamp(),
            )

            def _request(_timeout=None):
                model = self._message_helper.weights_to_model_proto(weights)
                train_results = controller_pb2.TrainResults(
                    metrics=metrics,
                    metadata=metadata
                )

                request = controller_pb2.TrainDoneRequest(
                    task=task,
                    model=model,
                    results=train_results,
                )

                return stub.TrainDone(
                    request=request,
                    timeout=_timeout
                )

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self):
        """Shuts down the client."""
        self._get_client()[2]()

    def _join_federation(
        self,
        stub: controller_pb2_grpc.ControllerServiceStub,
        request: controller_pb2.Learner,
        timeout: Optional[int] = None
    ) -> None:
        """Sends a request to the Controller to join the federation and stores the assigned learner id.

        Parameters
        ----------
        stub : controller_pb2_grpc.ControllerServiceStub
            The gRPC stub.
        request : controller_pb2.Learner
            The request Proto object with the Learner.
        timeout : Optional[int], optional
            The timeout in seconds, by default None

        """
        try:
            response = stub.JoinFederation(request, timeout=timeout)
            learner_id = response.id

            open(self._learner_id_fp, "w+").write(learner_id.strip())
            self._learner_id = learner_id

            MetisLogger.info(
                "Joined federation with learner id: {}".format(learner_id))
        except grpc.RpcError as rpc_error:

            if rpc_error.code() == grpc.StatusCode.ALREADY_EXISTS:

                learner_id = open(self._learner_id_fp, "r").read().strip()
                self._learner_id = learner_id
                MetisLogger.info(
                    "Rejoined federation with learner id: {}".format(learner_id))

            else:
                # FIXME: figure out how to handle this error
                MetisLogger.fatal("Unhandled grpc error: {}".format(rpc_error))

    def _has_learner_id(self) -> bool:
        """Checks if the learner id exists.

        Returns
        -------
        bool
            True if the learner id exists, False otherwise.
        """
        return self._learner_id is not None
