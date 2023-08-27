
"""A gRPC client used from the driver to communicate with the controller."""

from typing import Optional
from ..common.client import get_client
from ..proto import controller_pb2, controller_pb2_grpc, model_pb2, service_common_pb2
from ..common.types import ClientParams


class GRPCControllerClient(object):
    """A gRPC client used from the driver to communicate with the controller."""

    def __init__(
        self,
        client_params: ClientParams,
        max_workers=1
    ):
        """Initializes the client.

        Parameters
        ----------
        client_params : ClientParams
            The parameters needed to connect to the Controller.
        max_workers : int, optional
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._client_params = client_params
        self._max_workers = max_workers

    def _get_client(self):
        return get_client(
            stub_class=controller_pb2_grpc.ControllerServiceStub,
            client_params=self._client_params,
            max_workers=self._max_workers
        )

    def check_health_status(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.HealthStatusResponse:
        """Checks the health status of the controller.

        Parameters
        ----------
        request_retries : Optional[int], (default=1)
            The number of retries, by default 1
        request_timeout : Optional[int], (default=None)
            The timeout in seconds, by default None
        block : Optional[bool], (default=True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.HealthStatusResponse
            The response Proto object with the health status from the controller.
        """

        with self._get_client() as client:
            stub, schedule, _ = client

            return schedule(stub.GetHealthStatus,
                            request_retries, request_timeout, block)

    def set_initial_model(
        self,
        model: model_pb2.Model,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.Ack:
        """Sends an initial model to the Controller.

        Parameters
        ----------
        model : model_pb2.Model
            The initial model.
        request_retries : Optional[int], (default=1)
            The number of retries, by default 1
        request_timeout : Optional[int], (default=None)
            The timeout in seconds, by default None
        block : Optional[bool], (default=True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the ack from the controller.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.SetInitialModel(model, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def start_training(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.Ack:
        """Starts the federated training.

        Parameters
        ----------
        request_retries : Optional[int], optional
            The number of retries, by default 1
        request_timeout : Optional[int], optional
            The timeout in seconds, by default None
        block : Optional[bool], optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the ack from the controller.
        """

        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.StartTraining(service_common_pb2.StartTrainingRequest(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def get_statistics(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> controller_pb2.GetLogs:
        """Gets logs from the controller.

        Parameters
        ----------
        request_retries : Optional[int], (default=1)
            The number of retries, by default 1
        request_timeout : Optional[int], (default=None)
            The timeout in seconds, by default None
        block : Optional[bool], (default=True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        controller_pb2.GetStatisticsResponse
            The response Proto object with the statistics from the controller.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            # FIXME: convert to getLogs
            def _request(_timeout=None):
                request = service_common_pb2.Empty()
                return stub.GetStatistics(request, timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_server(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> service_common_pb2.Ack:
        """Sends a shutdown request to the controller.

        Parameters
        ----------
        request_retries : Optional[int], (default=1)
            The number of retries, by default 1
        request_timeout : Optional[int], (default=None)
            The timeout in seconds, by default None
        block : Optional[bool], (default=True)
            Whether to block until the request is completed, by default True

        Returns
        -------
        service_common_pb2.Ack
            The response Proto object with the ack from the controller.

        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                return stub.ShutDown(service_common_pb2.ShutDownRequest(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self):
        """Shuts down the client."""

        with self._get_client() as client:
            _, _, shutdown = client
            shutdown()
