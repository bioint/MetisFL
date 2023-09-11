import json
from typing import Any, Dict, List, Optional

import grpc
import numpy as np
from loguru import logger

from metisfl.common.client import get_client
from metisfl.common.formatting import get_timestamp
from metisfl.common.types import ClientParams
from metisfl.learner.message import MessageHelper
from metisfl.proto import (controller_pb2, controller_pb2_grpc, learner_pb2,
                           service_common_pb2)


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
        max_workers: Optional[int] = 1
    ):
        """A gRPC client used from the Learner to communicate with the Controller.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters. Used by the learner to connect to the Controller.
        message_helper : MessageHelper
            The MessageHelper object used to serialize/deserialize the messages.
        max_workers : Optional[int], (default: 1)
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self.client_params = client_params
        self.message_helper = message_helper
        self.max_workers = max_workers

        # Must be initialized after joining the federation
        self.learner_id = None

    def get_client(self):
        return get_client(
            stub_class=controller_pb2_grpc.ControllerServiceStub,
            client_params=self.client_params,
            max_workers=self.max_workers
        )

    def join_federation(
        self,
        client_params: ClientParams,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> None:
        """Sends a request to the controller to join the federation.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters of the Learner server. They are sent to the Controller 
            when joining the federation so that the Controller can connect to the Learner Server.
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True
        """

        with self.get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):
                request = controller_pb2.Learner(
                    hostname=client_params.hostname,
                    port=client_params.port,
                    root_certificate_bytes=read_certificate(
                        client_params.root_certificate),
                )
                response = stub.JoinFederation(request, timeout=_timeout)
                self.learner_id = response.id
                logger.success(
                    "Joined federation with learner id: {}".format(self.learner_id))

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

        if not self.has_learner_id():
            logger.warning(
                "Cannot leave federation before joining the federation.")
            return

        with self.get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):
                request = controller_pb2.LearnerId(
                    id=self.learner_id,
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

        if not self.has_learner_id():
            raise RuntimeError(
                "Cannot send train done before joining the federation.")

        logger.success("Completed training task with id: {}".format(task.id))

        with self.get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule = client[1]

            task = learner_pb2.Task(
                id=task.id,
                sent_at=task.sent_at,
                received_at=task.received_at,
                completed_at=get_timestamp(),
            )

            def _request(_timeout=None):
                model = self.message_helper.weights_to_model_proto(weights)

                train_results = controller_pb2.TrainResults(
                    metrics=json.dumps(metrics),
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
        with self.get_client() as client:
            _, _, shutdown = client
            shutdown()

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
            self.learner_id = response.id
            logger.info(
                "Joined federation with learner id: {}".format(learner_id))
        except grpc.RpcError as rpc_error:

            if rpc_error.code() == grpc.StatusCode.ALREADY_EXISTS:

                learner_id = open(self.learner_id_fp, "r").read().strip()
                self.learner_id = learner_id
                logger.info(
                    "Rejoined federation with learner id: {}".format(learner_id))

            else:
                # FIXME: figure out how to handle this error
                logger.critical("Unhandled grpc error: {}".format(rpc_error))

    def has_learner_id(self) -> bool:
        """Checks if the learner id exists.

        Returns
        -------
        bool
            True if the learner id exists, False otherwise.
        """
        return self.learner_id is not None
