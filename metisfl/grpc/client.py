"""gRPC client for the Metis Controller."""

import queue
import time
from ast import Tuple
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import grpc
from pebble import ThreadPool


from ..proto import controller_pb2_grpc
from ..utils.logger import MetisLogger
from .common import GRPC_MAX_MESSAGE_LENGTH, get_endpoint


def create_channel(
    server_address: str,
    root_certificate: Optional[Union[str, bytes]] = None,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> grpc.Channel:
    """Creates a gRPC channel to the given server address using the given root certificate.

    Parameters
    ----------
    server_address : str
        The server address in the form of "hostname:port".
    root_certificate : Optional[Union[str, bytes]], optional
        The root certificate, either as a string or bytes, by default None. If None, the connection is insecure.
    max_message_length : int, optional
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
            root_certificate)
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=options
        )
        MetisLogger.info(
            "Opened secure gRPC connection using given certificates")
    else:
        channel = grpc.insecure_channel(server_address, options=options)
        MetisLogger.info(
            "Opened insecure gRPC connection (no certificates given)")

    return channel


@contextmanager
def get_client(
    server_hostname: str,
    server_port: int,
    stub_class: Callable,
    root_certificate: Optional[str] = None,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    max_workers=1
) -> Iterator[
    Tuple[
        controller_pb2_grpc.ControllerServiceStub,
        Callable,
        Callable
    ]
]:
    """Gets a gRPC client for the given server hostname/port and stub class.

    Parameters
    ----------
    server_hostname : str
         The server hostname.
    server_port : int
        The server port.
    stub_class : Callable
        The stub class to be used with the created channel to establish a connection.
    root_certificate : Optional[str], optional
        The file path to the root certificate, by default None. If None, the connection is insecure.
    max_message_length : int, optional
        The maximum message length, by default GRPC_MAX_MESSAGE_LENGTH
    max_workers : int, optional
        The maximum number of workers for the client ThreadPool, by default 1

    Yields
    ------
    Iterator[ Tuple[ controller_pb2_grpc.ControllerServiceStub, Callable, Callable ] ]
        A tuple containing the stub, the schedule_request function and the shutdown function.
    """

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
        request_timeout : _type_, optional
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
    request_timeout: float,
    request_retries: int
) -> Any:
    """Sends a request to the controller with a timeout and retries.

    Parameters
    ----------
    request_fn : Callable
        A function that sends a request to the controller.
    request_timeout : float
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
            MetisLogger.info(
                "Exception Raised: {}, Retrying...".format(rpc_error))
            if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                time.sleep(10)
        else:
            break
        count_retries += 1
    return response
