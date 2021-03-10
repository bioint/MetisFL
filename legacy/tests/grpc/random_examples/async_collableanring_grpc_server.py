import sys
import os.path

## Set path so that you can import the local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from concurrent import futures
import threading
import grpc
import time


import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2_grpc as async_collaborative_learning_no_cache_pb2_grpc
import tests.grpc.random_examples.async_collaborative_learning_no_cache_pb2 as async_collaborative_learning_no_cache_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_ALL_CLIENTS = ('A', 'B', 'C')


class PendingUpdateRequest(object):

	def __init__(self, client_id, remaining_clients=list(), community_weights=dict(), control_ts=0):
		self.client_id = client_id
		self.control_ts = control_ts
		self.remaining_clients = remaining_clients
		self.community_weights = community_weights

	def is_community_ready(self):
		return len(self.remaining_clients) == 0

	def __str__(self):
		return "Pending Request of Client ID: {}\n\tControl TS: {},\n\tRemaining Clients: {},\n\tCommunity Weights: {},\n\tIs Community Ready: {}\n".format(
			self.client_id, self.control_ts, self.remaining_clients, self.community_weights, self.is_community_ready()
		)


class AsyncCollaborativeLearningHandler(async_collaborative_learning_no_cache_pb2_grpc.AsyncCollaborativeLearningServicer):

	def __init__(self, system_clients):
		self.system_clients = system_clients
		self.control_ts = 0
		self.pending_update_requests = dict()
		self.need_updated_gradients_from = dict()
		self.is_in_update_state = dict()
		self.in_update_state_gradients = dict()
		for client_id in self.system_clients:
			self.need_updated_gradients_from[client_id] = False
			self.is_in_update_state[client_id] = False
			self.in_update_state_gradients[client_id] = list()

	def __str__(self):

		pending_requests_str = '\n'.join([str(req) for req in self.pending_update_requests.values()])
		return "System Clients: {}\nCurrent TS: {}\nPending Update Requests: {}\nNeed Updated Gradients From: {}\nIs in Update State: {}\nUpdate State Weights: {}\n".format(
			self.system_clients, self.control_ts, pending_requests_str, self.need_updated_gradients_from, self.is_in_update_state, self.in_update_state_gradients)


	def StartShippingGradients(self, request, context):

		# Get the id of the requester
		requester_id = next(request).learner_id
		# check if client exists and whether its' latest values are needed
		# if yes return ack.status:True, else return ack.status:False
		ack = async_collaborative_learning_no_cache_pb2.Ack(status=False)
		if requester_id in self.need_updated_gradients_from:
			start_shipping = self.need_updated_gradients_from[requester_id]
			ack.status = start_shipping

		# print("-------------- StartShippingGradients --------------")
		# print("Requester:", requester_id)
		print(requester_id)
		# print(self)

		yield ack


	def DispatchGradients(self, request_iterator, context):
		# Process first request and store client's gradients
		first_request = next(request_iterator)
		requester_id = first_request.learner.learner_id

		# Update controller that we do not need gradients for now
		self.need_updated_gradients_from[requester_id] = False

		# Create requester's gradients collection and generate an acknowledgment value
		requester_gradients = list()
		requester_gradients.append(first_request.gradients.value)
		for request in request_iterator:
			request_gradients = request.gradients.value
			# TODO UNPICKLE GRADIENTS/FEDERATION VARIABLE OBJECT
			requester_gradients.append(request_gradients)
			ack = async_collaborative_learning_no_cache_pb2.Ack(status=True)
			# for each parsed request send an acknowledgment back to the requester
			yield ack

		# Iterate through all pending requests and if the requester is in the remaining clients list for a community update
		# then update pending request's community weight.
		for pending_request in self.pending_update_requests.values():
			if requester_id in pending_request.remaining_clients:
				pending_request.community_weights[requester_id] = requester_gradients
				pending_request.remaining_clients.remove(requester_id)

		print("-------------- DispatchGradients --------------")
		print("Requester:", requester_id)
		print(self)


	def RequestGradientsUpdate(self, request_iterator, context):

		# Process first request and store client's gradients
		first_request = next(request_iterator)
		requester_id = first_request.learner.learner_id

		# Sanity check. Do not allow a new update request for an existing requester
		if requester_id in self.pending_update_requests.keys():
			ack = async_collaborative_learning_no_cache_pb2.Ack(status=False)
			# return generator since the reply is a stream
			yield ack
			return

		self.control_ts += 1
		gradients = first_request.gradients.value
		self.is_in_update_state[requester_id] = True
		self.in_update_state_gradients[requester_id] = list()
		self.in_update_state_gradients[requester_id].append(gradients)

		# Save the gradients of the requesting client shipped with every request
		for request in request_iterator:
			request_gradients = request.gradients.value
			# TODO UNPICKLE GRADIENTS/FEDERATION VARIABLE OBJECT
			self.in_update_state_gradients[requester_id].append(request_gradients)
			ack = async_collaborative_learning_no_cache_pb2.Ack(status=True)
			# for each parsed request send an acknowledgment back to the requester
			yield ack

		# Update who we need new gradients from (of course not from the requester)
		self.need_updated_gradients_from[requester_id] = False
		requester_remaining_clients = list()
		for client_id in self.system_clients:
			if not self.is_in_update_state[client_id]:
				self.need_updated_gradients_from[client_id] = True
				requester_remaining_clients.append(client_id)


		# Update client's community gradients using gradients from clients that are still in update state
		requester_community_weights = dict()
		for client_id in self.is_in_update_state.keys():
			requester_community_weights[client_id] = self.in_update_state_gradients[client_id]

		# Create a new pending update request object
		pending_update_request = PendingUpdateRequest(client_id=requester_id,
													  remaining_clients=requester_remaining_clients,
													  community_weights=requester_community_weights,
													  control_ts=self.control_ts)

		# Update Controller state
		self.pending_update_requests[requester_id] = pending_update_request

		print("-------------- RequestGradientsUpdate --------------")
		print("Requester:", requester_id)
		print(self)


	def FetchUpdatedGradients(self, request, context):
		# Get the id of the requester
		requester_id = request.learner_id

		# If the requester has a pending request proceed
		if requester_id in self.pending_update_requests:
			# Retrieve requester's pending update request
			requester_pending_request = self.pending_update_requests[requester_id]
			is_community_ready = requester_pending_request.is_community_ready()
			# If community is ready stream True and the computed community gradients back to the client
			if is_community_ready:
				# TODO COMPUTE COMMUNITY AND PICKLE COMMUNITY GRADIENTS
				ack = async_collaborative_learning_no_cache_pb2.Ack(status=is_community_ready)
				gradients = async_collaborative_learning_no_cache_pb2.Gradients(value=str(requester_pending_request.community_weights[requester_id]))
				community_ready_response = async_collaborative_learning_no_cache_pb2.CommunityReadyResponse(ack=ack, gradients=gradients)
				yield community_ready_response

				# Update Controller's state
				self.is_in_update_state[requester_id] = False # requester is no longer in update state
				self.in_update_state_gradients[requester_id] = list() # erase requester's cached update state gradients
				del self.pending_update_requests[requester_id] # remove requester's update request

				print("-------------- FetchUpdatedGradients --------------")
				print("Requester:", requester_id)
				print(self)

				return

		# If the requester is not in update state or the community is not ready yet, just reply False with Gradients='-1'
		ack = async_collaborative_learning_no_cache_pb2.Ack(status=False)
		gradients = async_collaborative_learning_no_cache_pb2.Gradients(value="-1")
		community_ready_response = async_collaborative_learning_no_cache_pb2.CommunityReadyResponse(ack=ack, gradients=gradients)
		yield community_ready_response

		print("-------------- FetchUpdatedGradients --------------")
		print("Requester:", requester_id)
		print(self)


	def FetchFirstUpdatedGradients(self, request, context):

		update_no = 1
		time.sleep(10)
		while True:
			ack = async_collaborative_learning_no_cache_pb2.Ack(status=True)
			grad = "{}_{}.{}".format('Update Grads', update_no, time.time())
			gradients = async_collaborative_learning_no_cache_pb2.Gradients(value=str(grad))
			community_ready_response = async_collaborative_learning_no_cache_pb2.CommunityReadyResponse(ack=ack, gradients=gradients)
			yield community_ready_response
			update_no += 1
			time.sleep(10)


	def HelloWorld(self, request, context):
		return async_collaborative_learning_no_cache_pb2.HelloReply(value='Hello, %s!' % request.value)


def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	async_collaborative_learning_no_cache_pb2_grpc.add_AsyncCollaborativeLearningServicer_to_server(
		AsyncCollaborativeLearningHandler(system_clients=_ALL_CLIENTS),
		server=server)
	server.add_insecure_port('[::]:50051')
	server.start()

	try:
		while True:
			time.sleep(_ONE_DAY_IN_SECONDS)
	except KeyboardInterrupt:
		server.stop(0)

if __name__ == '__main__':

	server_thread = threading.Thread(target=serve)
	server_thread.start()

