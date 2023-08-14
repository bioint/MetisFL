
"""A gRPC client used from the Driver to communicate with the Learner."""

from typing import Optional
from metisfl.proto import learner_pb2_grpc

from metisfl.utils.fedenv import ClientParams
from ..proto import service_common_pb2
from ..grpc.client import get_client


class GRPCLearnerClient(object):

    """A gRPC client used from the Driver to communicate with the Learner."""

    def __init__(
        self,
        client_params: ClientParams,
        max_workers=1
    ):
        """Initializes the client.

        Parameters
        ----------
        server_hostname : str
            The hostname of the controller.
        server_port : int
            The port of the controller.
        root_certificate : Optional[str], optional
            The file path to the root certificate, by default None. If None, the connection is insecure.
        max_workers : int, optional
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

    def shutdown_learner(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block=False
    ) -> service_common_pb2.Ack:
        """Sends a shutdown request to the Learner.

        Parameters
        ----------
        request_retries : Optional[int], (default: 1)
            The number of retries for the request, by default 1
        request_timeout : Optional[int], (default: None)
            The timeout for the request, by default None
        block : bool, optional
            Whether to block until the request is completed, by default False

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the Ack.

        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):

                response = stub.ShutDown(
                    service_common_pb2.ShutDownRequest(),
                    timeout=_timeout)

                return response.ack.status

            return schedule(_request, request_retries, request_timeout, block)
