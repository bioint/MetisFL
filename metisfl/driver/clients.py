
"""The gRPC clients used from the Driver to communicate with the Controller and Learners."""

from typing import Callable, Optional
from metisfl.common.client import get_client
from metisfl.proto import controller_pb2, controller_pb2_grpc, learner_pb2_grpc, model_pb2, service_common_pb2
from metisfl.common.types import ClientParams


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
        self.client_params = client_params
        self.max_workers = max_workers

    def get_client(self):
        return get_client(
            stub_class=controller_pb2_grpc.ControllerServiceStub,
            client_params=self.client_params,
            max_workers=self.max_workers
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

        with self.get_client() as client:
            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule: Callable = client[1]

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
        with self.get_client() as client:
            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule: Callable = client[1]

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

        with self.get_client() as client:
            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule: Callable = client[1]

            def _request(_timeout=None):
                return stub.StartTraining(service_common_pb2.Empty(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def get_logs(
        self,
        request_retries: Optional[int] = 1,
        request_timeout: Optional[int] = None,
        block: Optional[bool] = True
    ) -> controller_pb2.Logs:
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
        with self.get_client() as client:

            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule: Callable = client[1]

            def _request(_timeout=None):
                request = service_common_pb2.Empty()
                return stub.GetLogs(request, timeout=_timeout)

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
        with self.get_client() as client:
            stub: controller_pb2_grpc.ControllerServiceStub = client[0]
            schedule: Callable = client[1]

            def _request(_timeout=None):
                return stub.ShutDown(service_common_pb2.Empty(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self):
        """Shuts down the client."""

        with self.get_client() as client:
            shutdown: Callable = client[2]
            shutdown()


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
        self.client_params = client_params
        self.max_workers = max_workers

    def get_client(self):
        return get_client(
            client_params=self.client_params,
            stub_class=learner_pb2_grpc.LearnerServiceStub,
            max_workers=self.max_workers
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
        with self.get_client() as client:
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
        with self.get_client() as client:
            stub: learner_pb2_grpc.LearnerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):
                return stub.SetInitialModel(model, timeout=_timeout)

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
        with self.get_client() as client:
            stub: learner_pb2_grpc.LearnerServiceStub = client[0]
            schedule = client[1]

            def _request(_timeout=None):
                return stub.ShutDown(service_common_pb2.Empty(), timeout=_timeout)

            return schedule(_request, request_retries, request_timeout, block)

    def shutdown_client(self) -> None:
        """Shuts down the client."""

        with self.get_client() as client:
            _, _, shutdown = client
            shutdown()
