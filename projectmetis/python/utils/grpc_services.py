import datetime
import grpc
import queue
import time

from projectmetis.python.logging.metis_logger import MetisLogger

from concurrent import futures
from grpc._cython import cygrpc
from pebble import ThreadPool
from projectmetis.proto.metis_pb2 import ServerEntity


class GRPCChannelMaxMsgLength(object):

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.host_port_str = "{}:{}".format(self.host, self.port)
        # TODO Remove this. Extend Channel class to read messages as chunks
        # TODO similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        self.channel_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
        self.channel = grpc.insecure_channel(target=self.host_port_str, options=self.channel_options)


class GRPCServerClient(object):

    def __init__(self, server_entity: ServerEntity, max_workers=1):
        self.executor = ThreadPool(max_workers=max_workers)
        self.executor_pool = queue.Queue()
        self._server_entity = server_entity
        self._channel = self.get_channel()

    def get_channel(self):
        """ Initialize connection only if it is not established. """
        _channel = GRPCChannelMaxMsgLength(
            self._server_entity.hostname,
            self._server_entity.port)
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

    def __init__(self, max_workers=1):
        # TODO Remove this. Extend Server class to read messages as chunks
        # TODO similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
        self.server_options = \
            [(cygrpc.ChannelArgKey.max_send_message_length, -1),
             (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.server = grpc.server(self.executor, options=self.server_options)
