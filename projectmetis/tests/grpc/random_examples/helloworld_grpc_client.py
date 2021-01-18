import sys
import os.path

## Set path so that you can import the local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import grpc
from tests.grpc.random_examples import helloworld_pb2, helloworld_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = helloworld_pb2_grpc.GreetingServiceStub(channel)
        response = stub.HelloWorld(helloworld_pb2.HelloRequest(value='you'))
    print("Greeter client received: " + response.value)


if __name__ == '__main__':
    run()