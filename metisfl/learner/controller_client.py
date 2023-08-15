from typing import Any, Dict, Optional

import grpc

from ..grpc.client import get_client
from ..proto import (controller_pb2, controller_pb2_grpc, model_pb2,
                     service_common_pb2)
from ..utils.fedenv import ClientParams
from ..utils.logger import MetisLogger


class GRPCControllerClient(object):

    def __init__(
        self,
        client_params: ClientParams,
        learner_id_fp: str,
        auth_token_fp: str,
        learner_cert_fp: Optional[str] = None,
        max_workers: Optional[int] = 1
    ):
        """A gRPC client used from the Learner to communicate with the Controller.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters. 
        learner_id_fp : str
            The file where the learner id is stored. 
        auth_token_fp : str
            The file where the auth token is stored.
        learner_cert_fp : Optional[str], (default: None)
            The learner certificate file path. This is sent to the controller when joining the federation
            to be used by the Controller gRPC client when connecting to the Learner server.
            If None; the connection to the Learner server is insecure.
            # TODO: clarify if this is the root or the public certificate.
        max_workers : Optional[int], (default: 1)
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._client_params = client_params
        self._learner_id_fp = learner_id_fp
        self._auth_token_fp = auth_token_fp
        self._learner_cert_fp = learner_cert_fp
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
        request_retries=1,
        request_timeout=None,
        block: bool = True
    ) -> controller_pb2.JoinFederationResponse:
        """Sends a request to the controller to join the federation.

        Parameters
        ----------
        num_training_examples : int
            The number of training examples of the local dataset.
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        controller_pb2.JoinFederationResponse
            The response Proto object with the Ack, the assigned learner id and the auth token.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                cert_bytes = open(
                    self._learner_cert_fp, "rb").read()

                request = controller_pb2.JoinFederationRequest(
                    hostname=self._server_params.hostname,
                    port=self._server_params.port,
                    public_certificate_bytes=cert_bytes,
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
            The response Proto object with the Ack from the Controller.
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
        metadata: Dict[str, str],
        epoch_evaluations: Dict[str, Any],
        aux_metadata: str = None,
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.Ack:
        """Sends the completed task to the Controller.

        Parameters
        ----------
        model : model_pb2.Model
            The model to be sent.
        metadata : Dict[str, str]
            The metadata to be sent.
        epoch_evaluations : Dict[str, Any]
            The epoch evaluations to be sent.
        aux_metadata : str, optional
            The auxiliary metadata to be sent, by default None
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack from the Controller.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):

                request = controller_pb2.TrainDoneRequest(
                    learner_id=self._learner_id,
                    auth_token=self._auth_token,
                    model=model,
                    metadata=metadata,
                    epoch_evaluations=epoch_evaluations,
                    aux_metadata=aux_metadata
                )

                return stub.MarkTaskCompleted(request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def _join_federation(
        self,
        stub: controller_pb2_grpc.ControllerServiceStub,
        request: controller_pb2.JoinFederationRequest,
        timeout=None
    ):
        try:
            response = stub.JoinFederation(request, timeout=timeout)
            learner_id, auth_token, status = \
                response.learner_id, response.auth_token, response.ack.status
            open(self._learner_id_fp, "w+").write(learner_id.strip())
            open(self._auth_token_fp, "w+").write(auth_token.strip())
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.ALREADY_EXISTS:
                learner_id = open(self._learner_id_fp, "r").read().strip()
                auth_token = open(self._auth_token_fp, "r").read().strip()
                status = True
            else:
                MetisLogger.fatal("Unhandled grpc error: {}".format(rpc_error))
        self._learner_id = learner_id
        self._auth_token = auth_token
        return learner_id, auth_token, status
