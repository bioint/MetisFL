import concurrent.futures
from pathlib import Path
import threading
from typing import Any, Callable, Union

import grpc
from metisfl.common.logger import MetisLogger

from metisfl.proto import controller_pb2_grpc, learner_pb2_grpc, service_common_pb2
from metisfl.common.types import ServerParams
from metisfl.common.formatting import get_endpoint, get_timestamp

GRPC_MAX_MESSAGE_LENGTH: int = 512 * 1024 * 1024


def get_server(
    server_params: ServerParams,
    servicer: Union[
        learner_pb2_grpc.LearnerServiceServicer,
        controller_pb2_grpc.ControllerServiceServicer
    ],
    add_servicer_to_server_fn: Callable,
    max_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> grpc.Server:
    """Creates a gRPC server using the given server parameters and servicer.

    Parameters
    ----------
    server_params : ServerParams
        The server configuration parameters.
    servicer: Union[learner_pb2_grpc.LearnerServiceServicer, controller_pb2_grpc.ControllerServiceServicer]
        The servicer for the gRPC server.
    add_servicer_to_server_fn: Callable
        The function to add the servicer to the server.
    max_workers : int, optional
        The maximum number of clients that can be handled concurrently, by default 1000
    max_message_length : int, optional
        The maximum message length, by default GRPC_MAX_MESSAGE_LENGTH

    Returns
    -------
    grpc.Server
        The gRPC server, not started.
    """
    server_hostname = server_params.hostname
    server_port = server_params.port

    endpoint = get_endpoint(server_hostname, server_port)

    options = [
        ("grpc.max_concurrent_streams", max_workers),
        ("grpc.max_send_message_length", max_message_length),
    ]

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_workers),
        options=options,
    )

    add_servicer_to_server_fn(servicer, server)

    if server_params.root_certificate is not None:
        root_certificate = Path(server_params.root_certificate).read_bytes()
        server_certificate = Path(
            server_params.server_certificate).read_bytes()
        private_key = Path(server_params.private_key).read_bytes()

        server_credentials = grpc.ssl_server_credentials(
            ((private_key, server_certificate),),
            root_certificates=root_certificate,
        )
        server.add_secure_port(endpoint, server_credentials)
    else:
        server.add_insecure_port(endpoint)

    return server


class Server:
    """Base class for the Controller and Learner servers. """

    server: grpc.Server = None
    status: service_common_pb2.ServingStatus = None
    shutdown_event: threading.Event = None

    def __init__(
        self,
        server_params: ServerParams,
        add_servicer_to_server_fn: Callable,
    ):
        """Initializes the server.

        Parameters
        ----------
        server_params : ServerParams
            The server configuration parameters.
        add_servicer_to_server_fn : Callable
            The function to add the servicer to the server.

        """

        self.status = service_common_pb2.ServingStatus.UNKNOWN
        self.shutdown_event = threading.Event()
        self.server_params = server_params

        self._server = get_server(
            server_params=server_params,
            servicer=self,
            add_servicer_to_server_fn=add_servicer_to_server_fn,
        )

    def start(self):
        """Starts the server. This is a blocking call and will block until the server is shutdown."""

        self._server.start()

        if self._server:
            self.status = service_common_pb2.ServingStatus.SERVING
            MetisLogger.info("Server started. Listening on: {}:{} with SSL: {}".format(
                self.server_params.hostname,
                self.server_params.port,
                "ENABLED" if self._is_ssl() else "DISABLED",
            ))
            self.shutdown_event.wait()
        else:
            # TODO: Should we raise an exception here?
            MetisLogger.error("Server failed to start.")

    def GetHealthStatus(self) -> service_common_pb2.HealthStatusResponse:
        """Returns the health status of the server."""

        return service_common_pb2.HealthStatusResponse(
            status=self.status,
        )

    def ShutDown(
        self,
        _: service_common_pb2.Empty,
        __: Any
    ) -> service_common_pb2.Ack:
        """Shuts down the server."""

        self.status = service_common_pb2.ServingStatus.NOT_SERVING
        self.shutdown_event.set()

        return service_common_pb2.Ack(
            status=True,
            timestamp=get_timestamp(),
        )

    def is_serving(self, context) -> bool:
        """Returns True if the server is serving, False otherwise."""

        if self.status != service_common_pb2.ServingStatus.SERVING:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            return False
        return True

    def is_ssl(self) -> bool:
        """Returns True if the server is using SSL, False otherwise."""

        return self.server_params.root_certificate is not None
