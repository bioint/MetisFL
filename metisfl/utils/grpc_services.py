import grpc
import queue
import time

from metisfl.utils.metis_logger import MetisLogger

from concurrent import futures
from grpc._cython import cygrpc
from pebble import ThreadPool
from metisfl.utils.ssl_configurator import SSLConfigurator
from metisfl.proto.metis_pb2 import ServerEntity


class GRPCEndpoint(object):

    def __init__(self, server_entity: ServerEntity):
        self.server_entity = server_entity
        self.listening_endpoint = "{}:{}".format(
            server_entity.hostname,
            server_entity.port)


class GRPCChannelMaxMsgLength(object):

    def __init__(self, server_entity: ServerEntity):
        self.grpc_endpoint = GRPCEndpoint(server_entity)
        # TODO Remove this. Extend Channel class to read messages as chunks
        # TODO similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        self.channel_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1)]

        public_certificate, _ = SSLConfigurator().load_certificates_from_ssl_config_pb(
            ssl_config_pb=server_entity.ssl_config, as_stream=True)
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


class GRPCServerClient(object):

    def __init__(self, server_entity: ServerEntity, max_workers=1):
        self.grpc_endpoint = GRPCEndpoint(server_entity)
        self.executor = ThreadPool(max_workers=max_workers)
        self.executor_pool = queue.Queue()
        self._channel = self.get_channel()

    def get_channel(self):
        """ Initialize connection only if it is not established. """
        _channel = GRPCChannelMaxMsgLength(self.grpc_endpoint.server_entity)
        return _channel.channel

    def request_with_timeout(self, request_fn, request_timeout, request_retries):
        count_retries = 0
        response = None
        while count_retries < request_retries:
            try:
                response = request_fn(request_timeout)
            except grpc.RpcError as rpc_error:
                MetisLogger.info("Exception Raised: {}, Retrying...".format(rpc_error))
                if rpc_error.code() == grpc.StatusCode.UNAVAILABLE:
                    time.sleep(10)  # sleep for 10secs in-between requests if server is Unavailable.
            else:
                break
            count_retries += 1
        return response

    def shutdown(self):
        self.executor.close()
        self.executor.join()
        self._channel.close()


class GRPCServerMaxMsgLength(object):

    def __init__(self, max_workers=None, server_entity: ServerEntity = None):
        self.grpc_endpoint = GRPCEndpoint(server_entity)

        # TODO Remove this. Extend Server class to read messages as chunks
        # TODO similar to this in Go: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        # (cygrpc.ChannelArgKey.max_concurrent_streams, 1000),
        # (grpc.chttp2.lookahead_bytes, 1024),
        # (grpc.chttp2.max_frame_size, 16777215)]
        self.channel_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1), ]
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.server = grpc.server(self.executor, options=self.channel_options)

        public_certificate, private_key = SSLConfigurator().load_certificates_from_ssl_config_pb(
            ssl_config_pb=server_entity.ssl_config, as_stream=True)
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
