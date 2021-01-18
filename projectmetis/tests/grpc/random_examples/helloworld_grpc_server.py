import sys
import os.path

## Set path so that you can import the local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import grpc
import time
from concurrent import futures
from tests.grpc.random_examples import helloworld_pb2, helloworld_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Greeter(helloworld_pb2_grpc.GreetingServiceServicer):

	def HelloWorld(self, request, context):
		return helloworld_pb2.HelloReply(value='Hello, %s!' % request.value)



def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	helloworld_pb2_grpc.add_GreetingServiceServicer_to_server(Greeter(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	try:
		while True:
			time.sleep(_ONE_DAY_IN_SECONDS)
	except KeyboardInterrupt:
		server.stop(0)


if __name__ == '__main__':
	serve()
