
"""A gRPC client used from the driver to communicate with the controller."""

from metisfl.grpc.client import get_client
from metisfl.proto import controller_pb2, controller_pb2_grpc, service_common_pb2
from metisfl.utils.fedenv import ClientParams


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
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.HealthStatusResponse:
        """Checks the health status of the controller.

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
        service_common_pb2.HealthStatusResponse
            The response Proto object with the health status from the controller.
        """

        with self._get_client() as client:
            stub, schedule, _ = client

            return schedule(stub.GetHealthStatus,
                            request_retries, request_timeout, block)

    def get_statistics(
        self,
        community_evaluation_backtracks: int,
        local_task_backtracks: int,
        metadata_backtracks: int,
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> controller_pb2.GetStatisticsResponse:
        """Gets statistics from the controller.

        Parameters
        ----------
        community_evaluation_backtracks : int
            The number of community evaluation backtracks.
        local_task_backtracks : int
            The number of local task backtracks.
        metadata_backtracks : int
            The number of metadata backtracks.
        request_retries : int, optional
            The number of retries, by default 1
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        controller_pb2.GetStatisticsResponse
            The response Proto object with the statistics from the controller.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request():
                request = controller_pb2.GetStatisticsRequest(
                    community_evaluation_backtracks=community_evaluation_backtracks,
                    local_task_backtracks=local_task_backtracks,
                    metadata_backtracks=metadata_backtracks
                )
                return stub.GetStatistics(request)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_controller(
        self,
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> service_common_pb2.Ack:
        """Sends a shutdown request to the controller.

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
            The response Proto object with the ack from the controller.
        """
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request():
                return stub.ShutDown(service_common_pb2.ShutDownRequest())

            return schedule(_request, request_retries, request_timeout, block)
