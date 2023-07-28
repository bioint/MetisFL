import grpc
import os
import queue
import time

from concurrent import futures
from grpc._cython import cygrpc
from pebble import ThreadPool

from metisfl.proto import controller_pb2_grpc, service_common_pb2
from metisfl.proto.metis_pb2 import ServerEntity, SSLConfig, SSLConfigFiles, SSLConfigStream
from metisfl.utils.logger import MetisLogger


class SSLProtoHelper(object):

    @classmethod
    def from_ssl_config_pb(cls, ssl_config_pb: SSLConfig, as_stream=False):
        public_certificate, private_key = None, None
        if ssl_config_pb.enable:            
            ssl_config_attr = getattr(ssl_config_pb, ssl_config_pb.WhichOneof('config'))
            # If the certificate is given then establish secure channel connection.
            if isinstance(ssl_config_attr, SSLConfigFiles):
                public_certificate = ssl_config_pb.ssl_config_files.public_certificate_file
                private_key = ssl_config_pb.ssl_config_files.private_key_file
                if as_stream:
                    public_certificate = cls.load_file_as_stream(public_certificate)
                    private_key = cls.load_file_as_stream(private_key)
            elif isinstance(ssl_config_attr, SSLConfigStream):
                public_certificate = ssl_config_pb.ssl_config_stream.public_certificate_stream
                private_key = ssl_config_pb.ssl_config_stream.private_key_stream
            else:
                MetisLogger.warning("Even though SSL was requested the certificate "
                                    "was not provided. Proceeding without SSL.")

        return public_certificate, private_key

    @classmethod
    def load_file_as_stream(cls, filepath):
        stream = None
        if filepath: 
            if os.path.exists(filepath):
                stream = open(filepath, "rb").read()
            else:
                MetisLogger.warning(
                    "The given filepath: {} does not exist.".format(filepath))
        return stream


class GRPCEndpoint(object):

    def __init__(self, server_entity_pb: ServerEntity):
        self.listening_endpoint = "{}:{}".format(
            server_entity_pb.hostname,
            server_entity_pb.port)


class GRPCChannelMaxMsgLength(object):

    def __init__(self, server_entity_pb: ServerEntity):
        self.grpc_endpoint = GRPCEndpoint(server_entity_pb)
        # TODO(@stripeli): Remove this. Extend Channel class to read messages as chunks
        #  similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        self.channel_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1)]

        # To initialize a grpc connection to a remote 
        # server using SSL we only need the public 
        # certificate not the private key.
        public_certificate, _ = \
            SSLProtoHelper.from_ssl_config_pb(
                server_entity_pb.ssl_config, 
                as_stream=True)
        if public_certificate:            
            ssl_credentials = grpc.ssl_channel_credentials(public_certificate)
            self.channel = grpc.secure_channel(
                target=self.grpc_endpoint.listening_endpoint,
                options=self.channel_options,
                credentials=ssl_credentials)
        else:
            self.channel = grpc.insecure_channel(
                target=self.grpc_endpoint.listening_endpoint,
                options=self.channel_options)


class GRPCClient(object):

    def __init__(self, server_entity_pb: ServerEntity, max_workers=1):
        self.grpc_endpoint = GRPCEndpoint(server_entity_pb)
        self.executor = ThreadPool(max_workers=max_workers)
        self.executor_pool = queue.Queue()
        self._channel = GRPCChannelMaxMsgLength(server_entity_pb).channel
        self._stub = controller_pb2_grpc.ControllerServiceStub(self._channel)

    def check_health_status(self, request_retries=1, request_timeout=None, block=True):
        def _request(_timeout=None):
            get_services_health_status_request_pb = service_common_pb2.GetServicesHealthStatusRequest()
            MetisLogger.info("Requesting controller's health status.")
            response = self._stub.GetServicesHealthStatus(
                get_services_health_status_request_pb, timeout=_timeout)
            MetisLogger.info("Received controller's health status, {} - {}".format(
                self.grpc_endpoint.listening_endpoint, response))
            return response
        return self.schedule_request(_request, request_retries, request_timeout, block)

    def schedule_request(self, request, request_retries=1, request_timeout=None, block=True):
        if request_retries > 1:
            future = self.executor.schedule(function=self._request_with_timeout,
                                            args=(request, request_timeout, request_retries))
        else:
            future = self.executor.schedule(request)

        if block:
            return future.result()
        else:
            self.executor_pool.put(future)

    def shutdown(self):
        self.executor.close()
        self.executor.join()
        self._channel.close()

    def _request_with_timeout(self, request_fn, request_timeout, request_retries):
        count_retries = 0
        response = None
        while count_retries < request_retries:
            try:
                response = request_fn(request_timeout)
            except grpc.RpcError as rpc_error:
                MetisLogger.info(
                    "Exception Raised: {}, Retrying...".format(rpc_error))
                if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                    # sleep for 10secs in-between requests if server is Unavailable.
                    time.sleep(10)
            else:
                break
            count_retries += 1
        return response


class GRPCServerMaxMsgLength(object):

    def __init__(self, max_workers=None, server_entity_pb: ServerEntity = None):
        self.grpc_endpoint = GRPCEndpoint(server_entity_pb)

        # TODO(stripeli): Remove this. Extend Channel class to read messages as chunks
        #  similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        # (cygrpc.ChannelArgKey.max_concurrent_streams, 1000),
        # (grpc.chttp2.lookahead_bytes, 1024),
        # (grpc.chttp2.max_frame_size, 16777215)]
        self.channel_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1), ]
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.server = grpc.server(self.executor, options=self.channel_options)

        public_certificate, private_key = \
            SSLProtoHelper.from_ssl_config_pb(
                server_entity_pb.ssl_config, as_stream=True)
        if public_certificate and private_key:
            server_credentials = grpc.ssl_server_credentials((
                (private_key, public_certificate, ),
            ))
            self.server.add_secure_port(
                self.grpc_endpoint.listening_endpoint,
                server_credentials)
        else:
            self.server.add_insecure_port(
                self.grpc_endpoint.listening_endpoint)
