
"""A gRPC client used from the Driver to communicate with the Learner."""

from typing import Optional
from ..proto import service_common_pb2
from ..grpc.client import get_client


class GRPCLearnerClient(object):

    """A gRPC client used from the Driver to communicate with the Learner."""
    
    def __init__(
        self,
        server_hostname: str,
        server_port: int,
        root_certificate: Optional[str] = None,
        max_workers=1
    ):
        """Initializes the client.

        Parameters
        ----------
        server_hostname : str
            The hostname of the controller. "localhost" if running locally.
        server_port : int
            The port of the controller.
        root_certificate : Optional[str], optional
            The file path to the root certificate, by default None. If None, the connection is insecure.
        max_workers : int, optional
            The maximum number of workers for the client ThreadPool, by default 1
        """
        self._server_hostname = server_hostname
        self._server_port = server_port
        self._root_certificate = root_certificate
        self._max_workers = max_workers
        
    def _get_client(self):
        return get_client(
            self._hostname,
            self._port,
            self._root_certificate,
            self._max_workers
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
