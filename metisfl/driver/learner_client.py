
"""A gRPC client used from the Driver to communicate with the Learner."""

from typing import Optional

from ..grpc.client import get_client
from ..proto import learner_pb2_grpc, model_pb2, service_common_pb2
from ..utils.fedenv import ClientParams


class GRPCLearnerClient(object):

    """A gRPC client used from the Driver to communicate with the Learner."""

    def __init__(
        self,
        client_params: ClientParams,
        max_workers: Optional[int] = 1
    ):
        """Initializes the client.

        Parameters
        ----------
        client_params : ClientParams
            The client parameters. Contains server hostname and port.
        max_workers : Optional[int], (default: 1)
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._client_params = client_params
        self._max_workers = max_workers

    def _get_client(self):
        return get_client(
            client_params=self._client_params,
            stub_class=learner_pb2_grpc.LearnerServiceStub,
            max_workers=self._max_workers
        )

    def get_model(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block=True
    ) -> model_pb2.Model:
        """Requests the Learner to send the current model.

        Parameters
        ----------
        request_retries : Optional[int], (default: 1)
            The number of retries for the request, by default 1
        request_timeout : Optional[int], (default: None)
            The timeout for the request, by default None
        block : Optional[bool], (default: True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.GetModel(service_common_pb2.Empty(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def set_initial_model(
        self,
        model: model_pb2.Model,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.Ack:
        """Requests the Learner to set the initial weights.

        Parameters
        ----------
        model : model_pb2.Model
            The initial model weights
        request_retries : Optional[int], (default: 1)
            The number of retries for the request, by default 1
        request_timeout : Optional[int], (default: None)
            The timeout for the request, by default None
        block : Optional[bool], (default: True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.SetInitialWeights(model, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_server(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block=False
    ) -> service_common_pb2.Ack:
        """Sends a shutdown request to the Learner server.

        Parameters
        ----------
        request_retries : Optional[int], (default: 1)
            The number of retries for the request, by default 1
        request_timeout : Optional[int], (default: None)
            The timeout for the request, by default None
        block : Optional[bool], (default: True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.ShutDown(service_common_pb2.Empty(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self) -> None:
        """Shuts down the client."""

        with self._get_client() as client:
            _, _, shutdown = client
            shutdown()
