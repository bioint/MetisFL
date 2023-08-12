


from ..proto import service_common_pb2
from ..utils.fedenv import ServerParams
from ..utils.logger import MetisLogger
from ..grpc.client import get_client


class GRPCLearnerClient(object):

    def __init__(
        self, 
        learner_server_params: ServerParams,
        max_workers=1
    ):
        self._server_params = learner_server_params
        self._max_workers = max_workers

        self._grpc_endpoint = self._learner_server_params.grpc_endpoint
        self._stub = None
        
    def _get_client(self):
        return get_client(
            self._server_params.hostname,
            self._server_params.port,
            self._server_params.root_certificate,
            max_workers=self._max_workers
        )
        
    def shutdown_learner(
        self, 
        request_retries=1, 
        request_timeout=None, 
        block=True
    ):
        with self._get_client() as client:
            stub, schedule, _ = client

            def _request(_timeout=None):
                
                response = stub.ShutDown(
                    service_common_pb2.ShutDownRequest(), 
                    timeout=_timeout)
                
                MetisLogger.info("Sent shutdown request to learner {}, response: {}.".format(
                    self._grpc_endpoint.listening_endpoint, response))
                
                return response.ack.status
            
            return schedule(_request, request_retries, request_timeout, block)
