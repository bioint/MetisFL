import grpc

import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2 as async_collaborative_learning_no_cache_pb2
import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2_grpc as async_collaborative_learning_no_cache_pb2_grpc


def generate_gradients(client_id, gradients):
	for gradient in gradients:
		# TODO PICKLE GRADIENTS OBJECT
		dispatch_request = async_collaborative_learning_no_cache_pb2.DispatchGradientsRequest(
			learner=async_collaborative_learning_no_cache_pb2.Learner(learner_id=client_id),
			gradients=async_collaborative_learning_no_cache_pb2.Gradients(value=gradient))
		yield dispatch_request


def start_shipping_gradients(stub, client_id):
	learner = async_collaborative_learning_no_cache_pb2.Learner(learner_id=client_id)
	ack = stub.StartShippingGradients(learner)
	print(ack.status)
	return ack.status


def dispatch_gradients(stub, client_id, gradients):
	# TODO PICKLE GRADIENTS OBJECT
	gradients_generator = generate_gradients(client_id, gradients)
	acks_iterator = stub.DispatchGradients(gradients_generator)
	for ack in acks_iterator:
		print(ack)


def request_gradients_update(stub, client_id, gradients):
	# TODO PICKLE GRADIENTS OBJECT
	gradients_generator = generate_gradients(client_id, gradients)
	acks_iterator = stub.RequestGradientsUpdate(gradients_generator)
	for ack in acks_iterator:
		print(ack)


def fetch_updated_gradients(stub, client_id):
	learner = async_collaborative_learning_no_cache_pb2.Learner(learner_id=client_id)
	community_ready_response_iterator = stub.FetchUpdatedGradients(learner)
	first_response = next(community_ready_response_iterator)
	is_community_ready = first_response.ack.status

	if is_community_ready:
		first_gradients = first_response.gradients.value
		print(first_gradients)
		for response in community_ready_response_iterator:
			next_gradients = response.gradients.value
			print(next_gradients)
		print('Update is ready')
	else:
		print('Update Not Ready')


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

	_ALL_CLIENTS = ('A', 'B', 'C')

	# Simulation A
	with grpc.insecure_channel('localhost:50051') as channel:
		CLIENT_ID = "A"
		GRADIENTS = ["{}_L{}".format(CLIENT_ID, i) for i in range(5)]
		stub = async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningStub(channel)
		print("-------------- StartShippingGradients --------------")
		start = start_shipping_gradients(stub, CLIENT_ID)
		print("-------------- DispatchGradients --------------")
		dispatch_gradients(stub, CLIENT_ID, GRADIENTS)
		print("-------------- RequestGradientsUpdate --------------")
		request_gradients_update(stub, CLIENT_ID, GRADIENTS)
		print("-------------- StartShippingGradients --------------")
		start_shipping_gradients(stub, CLIENT_ID)
		print("-------------- FetchUpdatedGradients --------------")
		fetch_updated_gradients(stub, CLIENT_ID)

	# Simulation B
	with grpc.insecure_channel('localhost:50051') as channel:
		CLIENT_ID = "B"
		GRADIENTS = ["{}_L{}".format(CLIENT_ID, i) for i in range(5)]
		stub = async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningStub(channel)
		print("-------------- StartShippingGradients --------------")
		start = start_shipping_gradients(stub, CLIENT_ID)
		print("-------------- DispatchGradients --------------")
		dispatch_gradients(stub, CLIENT_ID, GRADIENTS)
		print("-------------- RequestGradientsUpdate --------------")
		request_gradients_update(stub, CLIENT_ID, GRADIENTS)
		print("-------------- StartShippingGradients --------------")
		start_shipping_gradients(stub, CLIENT_ID)
		print("-------------- FetchUpdatedGradients --------------")
		fetch_updated_gradients(stub, CLIENT_ID)

	# Simulation C
	with grpc.insecure_channel('localhost:50051') as channel:
		CLIENT_ID = "C"
		GRADIENTS = ["{}_L{}".format(CLIENT_ID, i) for i in range(5)]
		stub = async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningStub(channel)
		print("-------------- StartShippingGradients --------------")
		start = start_shipping_gradients(stub, CLIENT_ID)
		if start:
			print("-------------- DispatchGradients --------------")
			dispatch_gradients(stub, CLIENT_ID, GRADIENTS)


# Simulation A
	with grpc.insecure_channel('localhost:50051') as channel:
		CLIENT_ID = "A"
		GRADIENTS = ["{}_L{}".format(CLIENT_ID, i) for i in range(5)]
		stub = async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningStub(channel)
		print("-------------- FetchUpdatedGradients --------------")
		fetch_updated_gradients(stub, CLIENT_ID)

if __name__ == '__main__':
    run()