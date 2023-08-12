import concurrent.futures
from ast import Tuple
from pathlib import Path
from typing import Callable, Optional, Union

import grpc

from ..learner.learner_servicer import LearnerServicer
from ..utils.fedenv import ServerParams
from .common import GRPC_MAX_MESSAGE_LENGTH, get_endpoint


def get_server(
    server_params: ServerParams,
    servicer_and_add_fn: Tuple[LearnerServicer, Callable],
    max_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> grpc.Server:
    """Creates a gRPC server, either secure or insecure."""

    server_hostname = server_params.hostname
    server_port = server_params.port
    servicer, add_servicer_to_server_fn = servicer_and_add_fn

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
        public_certificate = Path(
            server_params.public_certificate).read_bytes()
        private_key = Path(server_params.private_key).read_bytes()

        server_credentials = grpc.ssl_server_credentials(
            ((private_key, public_certificate),),
            root_certificates=root_certificate,
        )
        server.add_secure_port(endpoint, server_credentials)
    else:
        server.add_insecure_port(endpoint)

    return server
