import grpc

from grpc._cython import cygrpc
from concurrent import futures
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

	def __init__(self, server_entity: ServerEntity):
		self._server_entity = server_entity
		self._channel = self.get_channel()

	def get_channel(self):
		""" Initialize connection only if it is not established. """
		_channel = GRPCChannelMaxMsgLength(
			self._server_entity.hostname,
			self._server_entity.port)
		return _channel.channel


class GRPCServerMaxMsgLength(object):

	def __init__(self, max_workers=1):
		# TODO Remove this. Extend Server class to read messages as chunks
		# TODO similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
		self.server_options = \
			[(cygrpc.ChannelArgKey.max_send_message_length, -1),
			 (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
		self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
		self.server = grpc.server(self.executor, options=self.server_options)