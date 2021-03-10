import json

from concurrent.futures import ThreadPoolExecutor
from federation.fed_protobuff import collaborative_learning_pb2_grpc
from utils.devops.grpc_services import GRPCChannel
from utils.devops.proto_buff_exchange_ops import ProtoBuffExchangeOps
from utils.logging.metis_logger import MetisLogger as metis_logger


class FedClient(object):

	def __init__(self, client_id, controller_host_port, listen_for_federation_round_signals=False,
				 listen_for_session_termination_signals=False, thread_pool_workers=3, *args, **kwargs):

		self.client_id = client_id
		self.__channel = GRPCChannel(controller_host_port).channel
		self.__stub = collaborative_learning_pb2_grpc.CollabLearningStub(channel=self.__channel)
		self.__num_training_examples = 0
		self.__num_validation_examples = 0
		self.__trained_variables = list()
		self.__completed_epochs = 0
		self.__completed_batches = 0
		self.__batch_size = 0
		self.__latest_train_score = 0
		self.__latest_validation_score = 0
		self.__target_local_epochs_for_community_update = 1
		self.__completed_epochs_in_adaptive_mode = 0
		self.__client_community_updates_reception = 0
		self.__client_to_controller_update_requests = 0
		self.__client_to_controller_weights_transmissions = 0
		self.__processing_ms_per_epoch = 0
		self.__thread_executor = ThreadPoolExecutor(max_workers=thread_pool_workers,
													thread_name_prefix='PoolExecutorOf_{}'.format(self.client_id))
		self.__client_shutdown = False

		self.community_update_received = False
		self.__latest_community_update = list()

		# Public flag, this will trigger the learner to stop the training or to continue
		self.is_federation_round_signal_reached = False # for (semi)synchronous execution
		self.is_session_termination_signal_reached = False # for asynchronous execution

		if listen_for_federation_round_signals:
			self.__thread_executor.submit(self.__listen_for_federation_round_signals)

		if listen_for_session_termination_signals:
			self.__thread_executor.submit(self.__listen_for_session_termination_signals)


	def toJSON_representation(self):
		return {'grpc_client_id': self.client_id, 'grpc_client_community_updates_reception': self.__client_community_updates_reception,
				'grpc_client_to_controller_update_requests': self.__client_to_controller_update_requests, 'grpc_client_to_controller_weights_transmissions': self.__client_to_controller_weights_transmissions}



	def __listen_for_federation_round_signals(self):
		# Initialize connection with controller. Just send an empty message.
		empty_message_pb = ProtoBuffExchangeOps.construct_empty_message_pb()
		ack_pb_response_iterator = self.__stub.NotifyLearnersFederationRoundSignalReached(empty_message_pb)
		for ack_pb in ack_pb_response_iterator:
			if ack_pb.status is True:
				self.is_federation_round_signal_reached = True
				return
			if self.__client_shutdown: # stop thread
				return


	def __listen_for_session_termination_signals(self):
		# Initialize connection with controller. Just send an empty message.
		empty_message_pb = ProtoBuffExchangeOps.construct_empty_message_pb()
		ack_pb_response_iterator = self.__stub.NotifyLearnersSessionTerminationSignalReached(empty_message_pb)
		for ack_pb in ack_pb_response_iterator:
			if ack_pb.status is True:
				self.is_session_termination_signal_reached = True
				return
			if self.__client_shutdown: # stop thread
				return


	def update_num_training_examples(self, val=0):
		self.__num_training_examples = int(val)

	def update_num_validation_examples(self, val=0):
		self.__num_validation_examples = int(val)

	def update_trained_variables(self, new_variables=list()):
		self.__trained_variables = new_variables

	def update_completed_epochs(self, val=1):
		self.__completed_epochs = int(val)

	def update_completed_epochs_in_adaptive_mode(self, val=0):
		self.__completed_epochs_in_adaptive_mode = val

	def update_processing_ms_per_epoch(self, val=0):
		self.__processing_ms_per_epoch = val

	def update_completed_batches(self, val=1):
		self.__completed_batches = int(val)

	def update_batch_size(self, val=1):
		self.__batch_size = int(val)

	def update_latest_train_score(self, val=0):
		self.__latest_train_score = val

	def update_latest_validation_score(self, val=0):
		self.__latest_validation_score = val

	def update_target_local_epochs_for_community_update(self, val=0):
		self.__target_local_epochs_for_community_update = int(val)

	def is_target_stat_score_reached(self, score) -> bool:
		value_pb = ProtoBuffExchangeOps.construct_double_value_pb(val=score)
		ack_pb = self.__stub.IsSystemStatScoreReached(value_pb)
		return ack_pb.status


	def is_target_exec_time_reached(self, exec_time) -> bool:
		value_pb = ProtoBuffExchangeOps.construct_double_value_pb(val=exec_time)
		ack_pb = self.__stub.IsSystemExecutionTimeReached(value_pb)
		return ack_pb.status


	def is_eligible_for_community_update(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(self.client_id)
		ack_pb = self.__stub.IsLearnerCommunityUpdateSignalReached(learner_pb)
		return ack_pb.status


	def is_eligible_for_async_community_update(self, target_epochs=1):
		# TODO In the case of adaptive execution we need to subtract the number of epochs completing while in adaptive mode to be consistent
		return ((self.__completed_epochs - self.__completed_epochs_in_adaptive_mode) / target_epochs).is_integer()


	def retrieve_latest_fedround_controller_metadata(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(self.client_id)
		jsonstring_pb = self.__stub.RetrieveFederationRoundExecutionMetadataFromController(learner_pb)
		return json.loads(jsonstring_pb.value) # result should be in jsonLearnerValidationLossDiffRequest


	def notify_controller_to_reset(self):
		empty_msg_pb = ProtoBuffExchangeOps.construct_empty_message_pb()
		ack_pb = self.__stub.ResetControllerFederationRoundCollections(empty_msg_pb)
		return ack_pb.status


	def _generate_fedround_signal(self, finished_training, finished_epoch):
		fedround_signals_pb = ProtoBuffExchangeOps.contstruct_fedround_signal(learner_id=self.client_id,
																			  finished_epoch=finished_epoch,
																			  finished_training=finished_training)
		yield fedround_signals_pb


	def _notify_controller_for_fedround_signals(self, finished_training=False, finished_epoch=False):
		fedround_signals_generator = self._generate_fedround_signal(finished_training, finished_epoch)
		ack_pb = self.__stub.NotifyControllerToUpdateFederationRoundSignals(fedround_signals_generator)
		if ack_pb.status:
			metis_logger.info("Learner: {}, Successfully updated controller signals. Finished Training: {}, Finished Epoch:{}".format(self.client_id, finished_training, finished_epoch))
		else:
			metis_logger.info("Learner: {}, Could not update controller signals ".format(self.client_id))
		return ack_pb.status


	def notify_controller_for_fedround_signals(self, finished_training=False, finished_epoch=False, block=True):
		future = self.__thread_executor.submit(self._notify_controller_for_fedround_signals, finished_training, finished_epoch)
		if block:
			status = future.result()
			return status


	def _generate_learner_execution_result(self):
		learner_execution_result_pb = ProtoBuffExchangeOps.construct_learner_execution_result(learner_id=self.client_id,
																							  matrices=self.__trained_variables,
																							  num_training_examples=self.__num_training_examples,
																							  num_validation_examples=self.__num_validation_examples,
																							  latest_train_score=self.__latest_train_score,
																							  latest_validation_score=self.__latest_validation_score,
																							  comp_epochs=self.__completed_epochs,
																							  comp_batches=self.__completed_batches,
																							  batch_size=self.__batch_size,
																							  processing_ms_per_epoch=self.__processing_ms_per_epoch,
																							  target_local_epochs=self.__target_local_epochs_for_community_update)
		yield learner_execution_result_pb


	def _send_local_variables_to_controller(self):
		execution_result_generator = self._generate_learner_execution_result()
		ack_pb = self.__stub.SendLearnerExecutionResultToController(execution_result_generator)
		if ack_pb.status:
			metis_logger.info("Learner: {}, Successfully sent model to the controller".format(self.client_id))
		else:
			metis_logger.info("Learner: {}, Could not send model to the controller".format(self.client_id))
		return ack_pb.status


	def send_model_local_trained_variables_to_controller(self, block=True):
		self.__client_to_controller_weights_transmissions += 1
		future = self.__thread_executor.submit(self._send_local_variables_to_controller)
		if block:
			status = future.result()
			return status


	def _request_current_community(self, send_learner_state=True):
		if send_learner_state:
			self.__client_to_controller_weights_transmissions += 1
			execution_result_generator = self._generate_learner_execution_result()
			network_matrices_pb = self.__stub.CommunityUpdateWithLearnerState(execution_result_generator)
			metis_logger.info("Learner: {}, Successfully received community model from the controller".format(self.client_id))
			network_matrices_pb = network_matrices_pb.matrices
		else:
			learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=self.client_id)
			network_matrices_pb = self.__stub.CommunityUpdateCurrentState(learner_pb)
			network_matrices_pb = network_matrices_pb.matrices
		community_update = ProtoBuffExchangeOps.reconstruct_ndarrays_from_network_matrices_pb(network_matrices_pb)

		# Guarantee that the incoming community is non empty
		if len(community_update) > 0:
			self.__client_community_updates_reception += 1
			self.__latest_community_update = community_update
			self.community_update_received = True

		return community_update


	def request_current_community(self, send_learner_state=True, block=True):
		self.__client_to_controller_update_requests += 1
		future = self.__thread_executor.submit(self._request_current_community, send_learner_state)
		if block:
			community_update = future.result()
			return community_update


	def retrieve_latest_community_update_with_flag_toggle(self):
		community_update = self.__latest_community_update
		self.community_update_received = False # toggle community update flag
		return community_update


	def retrieve_latest_community_update(self):
		community_update = self.__latest_community_update
		return community_update


	def request_target_local_iterations_for_update(self):
		learner_execution_metadata = ProtoBuffExchangeOps.construct_learner_execution_metadata(learner_id=self.client_id,
																							   num_training_examples=self.__num_training_examples,
																							   num_validation_examples=self.__num_validation_examples,
																							   latest_train_score=self.__latest_train_score,
																							   latest_validation_score=self.__latest_validation_score,
																							   comp_epochs=self.__completed_epochs,
																							   comp_batches=self.__completed_batches,
																							   batch_size=self.__batch_size,
																							   processing_ms_per_epoch=self.__processing_ms_per_epoch,
																							   target_local_epochs=self.__target_local_epochs_for_community_update)
		integer_value_pb = self.__stub.LearnerLocalIterationsRequest(learner_execution_metadata)
		return integer_value_pb.value


	def request_community_and_learner_global_scalar_clock(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=self.client_id)
		community_state_pb = self.__stub.LearnerCommunityStateMetadata(learner_pb)
		global_update_scalar_clock = community_state_pb.global_update_scalar_clock
		learner_global_update_scalar_clock = community_state_pb.learner_global_update_scalar_clock
		return global_update_scalar_clock, learner_global_update_scalar_clock


	def request_community_and_learner_global_community_steps(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=self.client_id)
		community_state_pb = self.__stub.LearnerCommunityStateMetadata(learner_pb)
		global_community_steps = community_state_pb.global_community_steps
		learner_previous_global_community_steps = community_state_pb.learner_previous_global_community_steps
		return global_community_steps, learner_previous_global_community_steps


	def request_eligibility_for_community_request_based_on_staleness(self, validation_phase_stalenesses, current_update_steps):
		learner_staleness_metadata_pb = ProtoBuffExchangeOps.construct_learner_staleness_metadata(learner_id=self.client_id,
																								  validation_phase_stalenesses=validation_phase_stalenesses,
																								  current_update_steps=current_update_steps)
		ack_pb = self.__stub.LearnerStalenessCommunityRequestEligibility(learner_staleness_metadata_pb)
		is_eligible = ack_pb.status
		return is_eligible


	def request_current_global_epoch_id(self):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=self.client_id)
		integer_value_pb = self.__stub.RequestGlobalEpochID(learner_pb)
		return integer_value_pb.value


	def shutdown(self, wait=False):
		self.__client_shutdown = True # Client state flag, for background threads to return
		self.__channel.close()
		self.__thread_executor.shutdown(wait=wait) # wait=False: Non-Graceful shutdown

