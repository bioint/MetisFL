import grpc
import time
import threading
import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2 as async_collaborative_learning_no_cache_pb2
import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2_grpc as async_collaborative_learning_no_cache_pb2_grpc

def generate_learner(learner_id):
	learner = async_collaborative_learning_no_cache_pb2.Learner(learner_id=learner_id)
	yield learner

def start_shipping_gradients(stub, client_id):
	ack_sream = stub.StartShippingGradients(generate_learner(client_id))
	return next(ack_sream).status

def run(host_id, channel, federation_lock):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

	stub = async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningStub(channel)
	for i in range(100):
		CLIENT_ID = "{}.{}".format(host_id, i)
		# print("-------------- StartShippingGradients --------------")
		with federation_lock:
			start = start_shipping_gradients(stub, CLIENT_ID)
			print(CLIENT_ID)
			# print(start)
		time.sleep(0.05)


if __name__ == '__main__':

	LEARNERS = ('A', 'B', 'C', 'D', 'F', 'E')
	threads = list()
	channel = grpc.insecure_channel('localhost:50051')
	federation_lock = threading.Lock()
	for host_id in LEARNERS:
		t = threading.Thread(target=run,
							 name='host_training_{}'.format(host_id),
							 args=[host_id, channel, federation_lock],
							 daemon=True)
		threads.append(t)

	for p in threads:
		p.start()

	for p in threads:
		p.join()