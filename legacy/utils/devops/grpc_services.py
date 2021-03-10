import abc
import grpc

from grpc._cython import cygrpc
from concurrent import futures

_WEEK_IN_SECONDS = 7 * 24 * 60 * 60


class GRPCServer(abc.ABC):

	def __init__(self, grpc_servicer, max_workers=1, thread_pool_executor=None, service_lifetime=_WEEK_IN_SECONDS):
		# TODO Remove this. Extend Server class to read messages as chunks
		# TODO similar to this, C++: https://jbrandhorst.com/post/grpc-binary-blob-stream/
		self.grpc_servicer = grpc_servicer
		self.server_options = [(cygrpc.ChannelArgKey.max_send_message_length, -1),
							   (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
		# TODO
		#  max_workers default value is: (os.cpu_count() or 1) * 5
		#  check whether an existing pool is passed or create a new one
		if thread_pool_executor is not None:
			self.executor = thread_pool_executor
		else:
			self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
		self.server = grpc.server(self.executor, options=self.server_options)
		self.service_lifetime = service_lifetime

	@abc.abstractmethod
	def start(self):
		pass


class GRPCChannel(object):

	def __init__(self, host_port):
		self.channel_options = [(cygrpc.ChannelArgKey.max_send_message_length, -1),
								(cygrpc.ChannelArgKey.max_receive_message_length, -1)]
		self.channel = grpc.insecure_channel(target=host_port, options=self.channel_options)
