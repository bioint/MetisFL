from typing import Any, Dict, Optional

import grpc

from ..grpc.client import get_client
from ..proto import (controller_pb2, controller_pb2_grpc, model_pb2,
                     service_common_pb2)
from ..utils.fedenv import ClientParams, ServerParams
from ..utils.logger import MetisLogger


def read_certificate(fp: str) -> bytes:
    if fp is None:
        return None
    with open(fp, "rb") as f:
        return f.read()


class GRPCClient(object):

    def __init__(
        self,
        client_params: ClientParams,
        learner_id_fp: str,
        max_workers: Optional[int] = 1
    ):
        """A gRPC client used from the Learner to communicate with the Controller.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters. Used by the learner to connect to the Controller.
        learner_id_fp : str
            The file where the learner id is stored. 
        auth_token_fp : str
            The file where the auth token is stored.
        max_workers : Optional[int], (default: 1)
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._client_params = client_params
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
            stub, schedule, _ = client

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
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                request = controller_pb2.LeaveFederationRequest(
                    learner_id=self._learner_id,
                    auth_token=self._auth_token
                )
                return stub.LeaveFederation(
                    request=request,
                    timeout=_timeout
                )

            return schedule(_request, request_retries, request_timeout, block)

    def train_done(
        self,
        model: model_pb2.Model,
        metrics: Dict[str, Any],
        metadata: Dict[str, str],
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.Ack:
        """Sends the completed task to the Controller.

        Parameters
        ----------
        model : model_pb2.Model
            The model to be sent.
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
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):

                request = controller_pb2.TrainDoneRequest(
                    learner_id=self._learner_id,
                    auth_token=self._auth_token,
                    model=model,
                    metrics=metrics,
                    metadata=metadata
                )

                return stub.MarkTaskCompleted(request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def _join_federation(
        self,
        stub: controller_pb2_grpc.ControllerServiceStub,
        request: controller_pb2.Learner,
        timeout: Optional[int] = None
    ) -> None:
        """Sends a request to the Controller to join the federation.

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
                MetisLogger.fatal("Unhandled grpc error: {}".format(rpc_error))
        self._learner_id = learner_id
        return learner_id
