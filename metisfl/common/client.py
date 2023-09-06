"""gRPC client for the Metis Controller."""

import queue
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union

import grpc
from loguru import logger
from pebble import ThreadPool

from metisfl.common.types import ClientParams
from metisfl.common.formatting import get_endpoint

GRPC_MAX_MESSAGE_LENGTH: int = 512 * 1024 * 1024


def create_channel(
    server_address: str,
    root_certificate: Optional[Union[str, bytes]] = None,
    max_message_length: Optional[int] = GRPC_MAX_MESSAGE_LENGTH
) -> grpc.Channel:
    """Creates a gRPC channel to the given server address using the given root certificate.

    Parameters
    ----------
    server_address : str
        The server address in the form of "hostname:port".
    root_certificate : Optional[Union[str, bytes]], optional
        The root certificate, either as a string or bytes, by default None. 
        If None, the connection is insecure.
    max_message_length : Optional[int], optional
        The maximum message length, by default GRPC_MAX_MESSAGE_LENGTH

    Returns
    -------
    grpc.Channel
        The gRPC channel.
    """

    if isinstance(root_certificate, str):
        root_certificate = Path(root_certificate).read_bytes()

    options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if root_certificate is not None:
        ssl_channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificate,
        )
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=options
        )
    else:
        channel = grpc.insecure_channel(server_address, options=options)

    return channel


@contextmanager
def get_client(
    client_params: ClientParams,
    stub_class: Callable,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    max_workers=1
):
    """Gets a gRPC client for the given server hostname/port and stub class.

    Parameters
    ----------
    client_params : ClientParams
        The client parameters.    
    stub_class : Callable
        The stub class to be used with the created channel to establish a connection.
    max_message_length : int, optional
        The maximum message length, by default GRPC_MAX_MESSAGE_LENGTH
    max_workers : int, optional
        The maximum number of workers for the client ThreadPool, by default 1

    Yields
    ------
    Iterator[ Tuple[ controller_pb2_grpc.ControllerServiceStub, Callable, Callable ] ]
        A tuple containing the stub, the schedule_request function and the shutdown function.
    """

    server_hostname = client_params.hostname
    server_port = client_params.port
    root_certificate = client_params.root_certificate

    endpoint = get_endpoint(server_hostname, server_port)

    channel = create_channel(
        endpoint,
        root_certificate,
        max_message_length
    )
    stub = stub_class(channel)

    executor = ThreadPool(max_workers=max_workers)
    executor_pool = queue.Queue()

    def schedule(
        request,
        request_retries=1,
        request_timeout=None,
        block=True
    ) -> Union[Any, None]:
        """Schedule a request with the given parameters.

        Parameters
        ----------
        request : Any
            The request to be scheduled.
        request_retries : int, optional
            The number of retries, by default 1 
        request_timeout : int, optional
            The timeout in seconds, by default None
        block : bool, optional
            Whether to block until the request is completed, by default True

        Returns
        -------
        Union[Any, None]
            If the request is blocking, the response is returned. 
            If the request is non-blocking, None is returned.
        """
        future = executor.schedule(function=request_with_timeout,
                                   args=(request, request_timeout, request_retries))

        return future.result() if block else executor_pool.put(future)

    def shutdown():
        executor.close()
        executor.join()
        channel.close()

    try:
        yield (stub, schedule, shutdown)
    finally:
        shutdown()


def request_with_timeout(
    request_fn: Callable,
    request_timeout: int,
    request_retries: int
) -> Any:
    """Sends a request to the controller with a timeout and retries.

    Parameters
    ----------
    request_fn : Callable
        A function that sends a request to the controller.
    request_timeout : int
        The timeout in seconds.
    request_retries : int
        The number of retries.

    Returns
    -------
    Any
        The response. If the request fails, None is returned.
    """
    # FIXME: check the return logic, must not return None

    count_retries = 0
    response = None
    while count_retries < request_retries:
        try:
            response = request_fn(request_timeout)
        except grpc.RpcError as rpc_error:
            logger.error(
                f"Request to failed with {rpc_error.code()}: {rpc_error.details()}")
            if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                time.sleep(10)
        else:
            break
        count_retries += 1
    return response
