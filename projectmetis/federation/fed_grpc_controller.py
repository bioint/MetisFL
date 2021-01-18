import concurrent
import copy
import json
import logging
import operator
import os
import threading
import time
import traceback

import numpy as np

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from federation.fed_protobuff import collaborative_learning_pb2_grpc
from scipy.cluster import hierarchy
from utils.devops.proto_buff_exchange_ops import ProtoBuffExchangeOps
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.devops.grpc_services import GRPCServer
from utils.generic.time_ops import TimeUtil



_LOOPBACK_SLEEP_SECONDS = 30


class LearnerExecutionResult(object):

	def __init__(self, learner_id, num_training_examples=1, num_validation_examples=0, latest_train_score=0, latest_validation_score=0, completed_epochs=1, completed_batches=1,
				 batch_size=0, processing_ms_per_epoch=0, update_request_num=0, global_update_num=0, previous_global_update_num=None, target_local_epochs=None):
		self.learner_id = learner_id
		self.num_training_examples = num_training_examples
		self.num_validation_examples = num_validation_examples
		self.latest_train_score = latest_train_score
		self.latest_validation_score = latest_validation_score
		self.completed_epochs = completed_epochs
		self.completed_batches = completed_batches
		self.batch_size = batch_size
		self.processing_ms_per_epoch = processing_ms_per_epoch
		self.processing_ms_per_batch = -1 if self.batch_size == 0 else \
			np.divide(self.processing_ms_per_epoch, np.divide(self.num_training_examples, self.batch_size))
		self.current_update_request_num = update_request_num
		self.finished_training = False
		self.global_update_num = global_update_num
		self.previous_global_update_num = previous_global_update_num
		self.target_local_epochs = target_local_epochs


class LearnerExecutionResultWihMatrices(LearnerExecutionResult):

	def __init__(self, learner_id, network_matrices, num_training_examples=1, num_validation_examples=0, latest_train_score=0, latest_validation_score=0, completed_epochs=1, completed_batches=1,
				 batch_size=1, processing_ms_per_epoch=1, update_request_num=0, global_update_num=0, previous_global_update_num=None, target_local_epochs=5,
				 previous_completed_batches=1, previous_community_update_steps=1, current_community_update_steps=1):
		LearnerExecutionResult.__init__(self, learner_id=learner_id, num_training_examples=num_training_examples, num_validation_examples=num_validation_examples, latest_train_score=latest_train_score,
										latest_validation_score=latest_validation_score, completed_epochs=completed_epochs, completed_batches=completed_batches, batch_size=batch_size,
										processing_ms_per_epoch=processing_ms_per_epoch, update_request_num=update_request_num, global_update_num=global_update_num, previous_global_update_num=previous_global_update_num,
										target_local_epochs=target_local_epochs)

		self.network_matrices = network_matrices

		# Staleness will be defined in terms of current_community_batches-previous_community_batches+completed_batches
		self.previous_completed_batches = previous_completed_batches
		self.previous_community_update_steps = previous_community_update_steps
		self.current_community_update_steps = current_community_update_steps
		self.current_update_steps = self.completed_batches - self.previous_completed_batches
		self.current_staleness = self.current_community_update_steps - \
								 self.previous_community_update_steps + \
								 self.current_update_steps

		# JLA
		self.current_staleness = self.current_community_update_steps - (self.previous_community_update_steps + self.current_update_steps)
		if self.current_staleness <= 0:
			self.current_staleness = 1

		if self.learner_id == "FederationDriver" or 'FedAsyncPoly' in os.environ:
			self.learner_weight = 1
		else:

			# FedAvg
			# self.learner_weight = self.num_training_examples

			# FedAnnealing
			# self.learner_weight = np.divide(1, np.sqrt(self.global_update_num - self.previous_global_update_num + 1))
			# self.learner_weight = np.divide(1, np.sqrt(self.current_staleness))

			# FedValidation Score
			# self.learner_weight = latest_validation_score

			# FedValidation Score with Annealing Functions (e.g. 1/sqrt(LE/UR))
			# self.learner_weight = np.divide(latest_validation_score, np.sqrt(self.global_update_num - self.previous_global_update_num + 1))
			# self.learner_weight = np.divide(latest_validation_score, np.sqrt(np.divide(self.completed_epochs+1, self.current_update_request_num+1)))
			# self.learner_weight = np.divide(latest_validation_score, np.power(np.e, self.global_update_num - self.previous_global_update_num + 1))
			# self.learner_weight = np.divide(latest_validation_score, np.logging(self.global_update_num - self.previous_global_update_num + 1))
			# self.learner_weight = np.divide(self.num_training_examples, np.sqrt(self.global_update_num - self.previous_global_update_num + 1))

			# TODO Need to pass the community function here!

			if self.latest_validation_score == 0 or self.latest_validation_score == 0.0 or np.isnan(self.latest_validation_score):
				# Annealing Weight: SQRT(STALENESS) or LOG10(STALENESS)
				# self.learner_weight = np.divide(self.num_training_examples, np.log10(self.current_community_update_steps
				# 																	 - self.previous_community_update_steps
				# 																	 + self.current_update_steps))
				# metis_logger.info(msg="{} CONTROLLER SIDE WEIGHTING VALUE: {}".format(self.learner_id, self.learner_weight))

				self.learner_weight = self.num_training_examples
				# self.learner_weight = self.current_update_steps

				# # TODO IF NOT DVW THEN EITHER WEIGHT BASED ON NUMBER OF TRAINING EXAMPLES (FedAvg) OR BASED ON STALENESS (FedAnnealing)
				# self.learner_weight = np.divide(1, np.sqrt(self.current_staleness))
				# print("CONTROLLER Learner Weight:", self.learner_weight)

				# self.learner_weight = self.completed_batches
				# self.learner_weight = np.divide(self.num_training_examples, np.sqrt(self.global_update_num - self.previous_global_update_num + 1))
				# self.learner_weight = np.divide(self.num_training_examples, np.sqrt(self.completed_epochs))
			else:
				# Correction Step: If it is the first time the learner makes an update request discount him based on his staleness
				# if self.previous_community_update_steps == 0:
				# 	self.learner_weight = np.divide(self.num_training_examples, self.current_staleness) * self.latest_validation_score
				# else:
				# 	self.learner_weight = self.num_training_examples * self.latest_validation_score
				# self.learner_weight = self.num_training_examples * self.latest_validation_score
				self.learner_weight = self.latest_validation_score
				# self.learner_weight = self.num_training_examples


		self.weighted_network_matrices = [self.learner_weight * matrix for matrix in self.network_matrices]

	def __str__(self):
		return "Number of examples: {}, Number of completed epochs: {}, Number of completed batches: {}, Weighted Matrices: {}"\
			.format(self.num_training_examples, self.completed_epochs, self.completed_batches, self.weighted_network_matrices)


class LearnersExecutionStateMap(object):

	def __init__(self):
		self._FED_ASYNC_PREVIOUS_COMMUNITY = []
		self.state_map = dict()
		self.community_weighted_matrices = list()
		self.are_community_weighted_matrices_cached = False
		self.community_normalization_value = 0
		self.is_norm_value_cached = False
		self.global_update_request_num = 0
		self.global_steps_num = 0
		self.global_epoch_id = 0
		self.lock = threading.Lock()

		self.validation_phase_staleness_map = dict()


	def reset_community_state_network_map(self):
		self.state_map = dict()
		self.community_weighted_matrices = list()
		self.validation_phase_staleness_map = dict()
		self.are_community_weighted_matrices_cached = False
		self.community_normalization_value = 0
		self.is_norm_value_cached = False


	def cached_learners(self):
		return list(self.state_map.keys())


	def number_of_cached_learners(self):
		return len(self.state_map.keys())


	def learner_exists(self, learner_id):
		return learner_id in self.state_map


	def increase_global_request_num_by1(self):
		# When an update in the state map is issued,
		# we update the global counter of issued requests
		self.global_update_request_num += 1


	def retrieve_learner_update_request_num(self, learner_id):
		return self.state_map[learner_id].current_update_request_num


	def retrieve_current_global_update_request_num(self):
		return self.global_update_request_num


	def increase_global_steps_by_val(self, val):
		self.global_steps_num += val


	def retrieve_learner_previous_completed_batches(self, learner_id):
		return self.state_map[learner_id].completed_batches


	def retrieve_learner_previous_community_global_steps_num(self, learner_id):
		return self.state_map[learner_id].previous_community_update_steps


	def retrieve_learner_current_community_global_steps_num(self, learner_id):
		return self.state_map[learner_id].current_community_update_steps


	def retrieve_current_community_global_steps(self):
		return self.global_steps_num


	def get_current_community_update_requests_staleness(self):
		learners_staleness = dict()
		for learner_id in self.state_map:
			learners_staleness[learner_id] = \
				self.global_update_request_num - self.state_map[learner_id].global_update_num + 1
		return learners_staleness


	def get_current_community_update_steps_staleness(self, new_update_steps):
		learners_staleness = dict()
		for learner_id in self.state_map:
			learners_staleness[learner_id] = self.compute_learner_community_update_steps_staleness(
				learner_id=learner_id,
				new_update_steps=new_update_steps)
		return learners_staleness


	def compute_learner_community_update_steps_staleness(self, learner_id, new_update_steps=None):
		update_steps = 0 if new_update_steps is None else new_update_steps
		# learner_update_steps_staleness = self.state_map[learner_id].current_community_update_steps - \
		# 		self.state_map[learner_id].previous_community_update_steps + \
		# 		update_steps
		learner_update_steps_staleness = self.global_steps_num - \
				self.state_map[learner_id].current_community_update_steps + \
				update_steps

		return learner_update_steps_staleness


	def retrieve_current_global_epoch(self, num_system_learners):
		""" Global epoch is the minimum number of local epochs
		completed across the federation."""
		if self.global_epoch_id == 0:
			self.global_epoch_id = 1
		else:
			if self.number_of_cached_learners() != num_system_learners:
				return self.global_epoch_id
			else:
				self.global_epoch_id = int(min([self.state_map[learner_id].completed_epochs
												for learner_id in self.state_map.keys()]))
		return self.global_epoch_id


	def update_state_map(self, learner_exec_result):
		assert isinstance(learner_exec_result, LearnerExecutionResultWihMatrices)
		learner_id = learner_exec_result.learner_id
		self.state_map[learner_id] = learner_exec_result


	def retrieve_learner_state(self, learner_id):
		return self.state_map[learner_id]


	def retrieve_learner_global_update_num(self, learner_id):
		return self.state_map[learner_id].global_update_num


	def update_cached_community_weighted_matrices(self, community_weighted_matrices):
		self.community_weighted_matrices = community_weighted_matrices
		self.are_community_weighted_matrices_cached = True


	def retrieve_cached_community_weighted_matrices(self):
		if self.are_community_weighted_matrices_cached:
			return self.community_weighted_matrices
		else:
			community_weighted_matrices = self.compute_community_weighted_matrices()
			self.update_cached_community_weighted_matrices(community_weighted_matrices)
			return self.community_weighted_matrices


	def update_cached_normalization_value(self, normalization_value):
		self.community_normalization_value = normalization_value
		self.is_norm_value_cached = True


	def retrieve_cached_normalization_value(self):
		if self.is_norm_value_cached:
			return self.community_normalization_value
		else:
			compute_community_normalization_value = self.compute_community_normalization_value()
			self.update_cached_normalization_value(compute_community_normalization_value)
			return self.community_normalization_value


	def compute_community_normalization_value(self, cache_results=False):
		"""
		This function essentially computes and optionally caches the normalization factor of the community
			e.g. Σ(ε_k * n_k)
			With ε_k being the number of epochs trained by learner k and n_k the number of examples seen by learner k

		Args:
			cache_results: whether to cache normalization value result or not

		Returns:

		"""
		if self.number_of_cached_learners() > 0:
			learners_ids = self.cached_learners()
			normalization_value = 0
			for learner_id in learners_ids:
				normalization_value += self.retrieve_learner_state(learner_id).learner_weight
			if cache_results:
				self.update_cached_normalization_value(normalization_value)
			return normalization_value


	def compute_community_weighted_matrices(self, cache_results=False):
		"""
		This function computes and optionally caches the weighted average value of every weight in the network
			e.g. Σ n_k * W_k
				 With n_k being the number of training examples of learner k and W_k the weight matrices of learner k
		Args:
			cache_results: whether to cache the freshly computed weighted average weights

		Returns:

		"""
		learners_ids = self.cached_learners()
		if len(learners_ids) > 0:
			# Perform a deep copy of the weights of the first learner to ensure non-violation of references, and then add the weights of all subsequent learners
			community_weighted_matrices = copy.deepcopy(self.retrieve_learner_state(learners_ids[0]).weighted_network_matrices)
			for learner_id in learners_ids[1:]:
				learner_weighted_matrices = self.retrieve_learner_state(learner_id).weighted_network_matrices
				for idx, learner_w_matrix in enumerate(learner_weighted_matrices):
					community_weighted_matrices[idx] += learner_w_matrix
			if cache_results:
				self.update_cached_community_weighted_matrices(community_weighted_matrices)

			return community_weighted_matrices


	def build_final_community_network(self, norm_value, matrices):
		# TODO Need to catch error here when norm_value is None or 0
		norm_factor = 1 / norm_value
		community_network = list()
		for matrix in matrices:
			community_network.append(norm_factor * matrix)
		return community_network


	def compute_community_with_updated_learner_wcache(self, learner_exec_result):
		"""
		This function must be called only when if we cache previous computed results
		Args:
			learner_exec_result:

		Returns:

		"""
		if self.is_norm_value_cached is False:
			self.compute_community_normalization_value(cache_results=True)
		if self.are_community_weighted_matrices_cached is False:
			self.compute_community_weighted_matrices(cache_results=True)

		learner_id = learner_exec_result.learner_id
		new_learner_weight = learner_exec_result.learner_weight
		new_learner_weighted_network_matrices = learner_exec_result.weighted_network_matrices

		learner_state = self.retrieve_learner_state(learner_id)
		old_learner_weight = learner_state.learner_weight
		old_learner_weighted_network_matrices = learner_state.weighted_network_matrices

		# get cached normalization value
		community_normalization_value = self.retrieve_cached_normalization_value() - old_learner_weight + new_learner_weight
		self.update_cached_normalization_value(community_normalization_value)

		community_weighted_matrices = self.retrieve_cached_community_weighted_matrices()
		for idx, matrix in enumerate(community_weighted_matrices):
			previous_community_matrix_state = matrix
			previous_community_matrix_state -= old_learner_weighted_network_matrices[idx]  # remove from the community learner's old weighted matrix values
			previous_community_matrix_state += new_learner_weighted_network_matrices[idx]  # add to the community learner's new weighted matrix values
			community_weighted_matrices[idx] = previous_community_matrix_state
		self.update_cached_community_weighted_matrices(community_weighted_matrices)

		# Transaction completed, we have applied all required operations, update state map
		self.update_state_map(learner_exec_result)

		community_network = self.build_final_community_network(community_normalization_value, community_weighted_matrices)

		return community_network


	def compute_community_with_new_learner_wcache(self, learner_exec_result):

		assert isinstance(learner_exec_result, LearnerExecutionResultWihMatrices)

		if self.is_norm_value_cached is False:
			self.compute_community_normalization_value(cache_results=True)
		if self.are_community_weighted_matrices_cached is False:
			self.compute_community_weighted_matrices(cache_results=True)

		community_weighted_matrices = self.retrieve_cached_community_weighted_matrices()
		for idx, matrix in enumerate(community_weighted_matrices):
			new_matrix = matrix + learner_exec_result.weighted_network_matrices[idx] # add new matrix to the existing weighted average
			community_weighted_matrices[idx] = new_matrix # assign the newly computed weighted matrix
		self.update_cached_community_weighted_matrices(community_weighted_matrices)

		community_normalization_value = self.retrieve_cached_normalization_value()
		community_normalization_value += learner_exec_result.learner_weight # update normalization value with a new weighted value
		self.update_cached_normalization_value(community_normalization_value)

		# Transaction completed update state map
		self.update_state_map(learner_exec_result)

		community_network = self.build_final_community_network(community_normalization_value, community_weighted_matrices)
		return community_network


	def compute_community_without_cache(self):

		community_normalization_value = self.compute_community_normalization_value(cache_results=False)
		community_weighted_matrices = self.compute_community_weighted_matrices(cache_results=False)
		community_network = self.build_final_community_network(community_normalization_value, community_weighted_matrices)
		return community_network


	def compute_cached_community(self):

		community_normalization_value = self.retrieve_cached_normalization_value()
		community_weighted_matrices = self.retrieve_cached_community_weighted_matrices()
		community_network = self.build_final_community_network(community_normalization_value, community_weighted_matrices)
		return community_network


	def compute_community_network(self, learner_exec_result=None, using_cache=False):
		"""

		Args:
			learner_exec_result: The new or updated results of a learner
			using_cache: Whether to use the cached version, which means we will go with the optimization (True) or we compute form scratch (False)

		Returns:

		"""

		if 'FedAsyncPoly' in os.environ:
			print("INSIDE FedAsync COMMUNITY Scheme")

			if learner_exec_result is None:
				return self.compute_community_without_cache()

			if len(self._FED_ASYNC_PREVIOUS_COMMUNITY) == 0:
				self._FED_ASYNC_PREVIOUS_COMMUNITY = learner_exec_result.network_matrices
			else:
				previous_global_update_request = learner_exec_result.previous_global_update_num
				current_global_update_request = learner_exec_result.global_update_num
				current_update_request_staleness = current_global_update_request - previous_global_update_request + 1
				learner_contribution = np.multiply(0.5, np.divide(1, np.sqrt(current_update_request_staleness)))
				learner_weighted_network_matrices = [learner_contribution * matrix for matrix in learner_exec_result.network_matrices]
				community_weighted_network_matrices = [(1-learner_contribution) * matrix for matrix in self._FED_ASYNC_PREVIOUS_COMMUNITY]
				new_community = []
				for idx, learner_matrix in enumerate(learner_weighted_network_matrices):
					new_community.append(np.add(community_weighted_network_matrices[idx], learner_matrix))
				self._FED_ASYNC_PREVIOUS_COMMUNITY = new_community
			return self._FED_ASYNC_PREVIOUS_COMMUNITY
		else:
			if using_cache is False:
				return self.compute_community_without_cache()  # compute everything from scratch, no cache
			elif using_cache and learner_exec_result is None:
				return self.compute_cached_community() # compute the current cached community
			else:
				learner_id = learner_exec_result.learner_id
				if learner_id not in self.state_map:
					return self.compute_community_with_new_learner_wcache(learner_exec_result) # if the learner does not exist in the state map account for him and compute community using the cache
				else:
					return self.compute_community_with_updated_learner_wcache(learner_exec_result) # if the learner exists in the state map use cache accordingly and update state map at the end


class FedRoundSpotter(object):

	def __init__(self, system_learners):
		self.fedround_learners_execution_metadata = dict()
		self.system_learners = system_learners
		for learner_id in system_learners:
			self.fedround_learners_execution_metadata[learner_id] = LearnerExecutionResult(learner_id=learner_id)

		self.fedround_completed_learners = 0
		self.fedround_completed_epochs = 0
		self.fedround_termination_signal_reached = False
		self.session_termination_signal_reached = False
		self.lock = threading.Lock()

	def retrieve_fedround_completed_epochs(self):
		return self.fedround_completed_epochs

	def retrieve_fedround_completed_learners(self):
		return self.fedround_completed_learners

	def retrieve_fedround_learner_completed_epochs(self, learner_id):
		return self.fedround_learners_execution_metadata[learner_id].completed_epochs

	def retrieve_fedround_termination_signal_reached(self):
		return self.fedround_termination_signal_reached

	def update_fedround_termination_signal_reached(self, value):
		self.fedround_termination_signal_reached = value

	def update_fedround_global_completed_epochs(self, value):
		self.fedround_completed_epochs += value

	def update_fedround_global_completed_learners(self, value):
		self.fedround_completed_learners += value

	def update_fedround_learner_completed_epochs(self, learner_id, value):
		self.fedround_learners_execution_metadata[learner_id].completed_epochs += value

	def update_fedround_learner_finished_training(self, learner_id, value):
		self.fedround_learners_execution_metadata[learner_id].finished_training = value

	def compute_learner_target_local_iterations(self, learner_id):
		"""
		:param learner_id:
		:return:
		"""
		tmax = -1
		for lid in self.fedround_learners_execution_metadata:
			learner_processing_ms_per_batch = self.fedround_learners_execution_metadata[lid].processing_ms_per_batch
			learner_training_examples = self.fedround_learners_execution_metadata[lid].num_training_examples
			learner_batch_size = self.fedround_learners_execution_metadata[lid].batch_size
			learner_total_batches = np.divide(learner_training_examples, learner_batch_size)
			learner_total_batches_ms = np.multiply(learner_total_batches, learner_processing_ms_per_batch)
			tmax = np.max([tmax, learner_total_batches_ms])

		# TODO Add Variable K that accounts for the number of local epochs for the slowest learner
		# tmax = np.multiple(k, tmax)
		requesting_learner_processing_ms_per_batch = self.fedround_learners_execution_metadata[learner_id]\
			.processing_ms_per_batch
		requesting_learner_local_iterations = np.divide(tmax, requesting_learner_processing_ms_per_batch)

		metis_logger.info("CONTROLLER MSG - LearnerID: {}, Target Local Iterations: {}"
						  .format(learner_id, requesting_learner_local_iterations))

		return requesting_learner_local_iterations

	def update_learner_target_local_epochs(self, learner_id, value):
		self.fedround_learners_execution_metadata[learner_id].target_local_epochs = value

	def __str__(self):
		return "Completed Learners: {}, Completed Epochs: {}".format(self.fedround_completed_learners, self.fedround_completed_epochs)

	def toJSON_representation(self):
		return {'fedround_completed_learners': self.fedround_completed_learners, 'fedround_completed_epochs': self.fedround_completed_epochs}


class ExecutionInterruptionSignals(object):

	def __init__(self, target_learners=None, target_epochs=None, target_score=None, target_exec_time_mins=None):
		self.target_learners = target_learners
		self.target_epochs = target_epochs
		self.target_score = target_score
		self.target_exec_time_mins = target_exec_time_mins
		self.lock = threading.Lock()

	def is_target_score_reached(self, acc_val):
		if self.target_score is not None:
			return acc_val >= self.target_score
		else:
			return False

	def is_execution_time_reached(self, current_exec_time_mins):
		if self.target_exec_time_mins is not None:
			return current_exec_time_mins >= self.target_exec_time_mins
		else:
			return False

	def is_target_learners_reached(self, current_learners):
		if self.target_learners is not None:
			return current_learners >= self.target_learners
		else:
			return False

	def is_target_epochs_reached(self, current_epochs, synchronous_execution=True):
		if synchronous_execution:
			if self.target_epochs is not None:
				return current_epochs >= self.target_epochs
			else:
				return False
		else:
			if self.target_epochs is not None:
				# We check whether the current epochs fully divide the target epochs
				# This condition is used in the asynchronous policy when a learner requests a community update
				eligible_update = (current_epochs / self.target_epochs).is_integer()
				return eligible_update
			else:
				return False


class SystemControllerServicer(collaborative_learning_pb2_grpc.CollabLearningServicer):

	def __init__(self, host_port, synchronous_execution, system_learners, target_learners, target_epochs, target_score, target_exec_time_mins, minimum_learners_for_community_update=3):
		self.host_port = host_port
		self.synchronous_execution = synchronous_execution
		self.semi_synchronous_execution = True if synchronous_execution is None else False
		self.system_learners = system_learners
		self.num_system_learners = len(self.system_learners)
		self.target_learners = target_learners
		# TODO Target update epochs for each learner must be done on a controller basis. As of now, the client is responsible to check for updates.
		self.target_epochs = target_epochs
		self.target_score = target_score
		self.target_exec_times_mins = target_exec_time_mins
		self.minimum_learners_for_community_update = minimum_learners_for_community_update
		# TODO Request Queue store
		# self.community_update_requests_q = queue.Queue()
		self.lock = threading.Lock()

		# Federation Round Specific
		self.current_federation_round = 1
		self.fedround_spotter = FedRoundSpotter(self.system_learners)
		self.fedround_learners_community_state = LearnersExecutionStateMap()
		self.fedexec_signals = ExecutionInterruptionSignals(target_learners=target_learners, target_epochs=target_epochs, target_score=target_score, target_exec_time_mins=target_exec_time_mins)

		self.controller_init_time = TimeUtil.current_milli_time()


		# TODO ONE TIME TRANSITION FLAG VARIABLE
		self.__transition_community_computation = False


	def IsSystemStatScoreReached(self, request, context):
		score = request.value

		try:
			with self.fedexec_signals.lock:
				is_reached = self.fedexec_signals.is_target_score_reached(score)
			# If we have asynchronous execution then we check on whether the target score is reached
			# This way we let all learners know that we have reached the end of the session
			with self.fedround_spotter.lock:
				self.fedround_spotter.update_fedround_termination_signal_reached(is_reached)

			ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=is_reached)
			return ack_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 1: {}".format(e))
			logging.error(traceback.format_exc())


	def IsSystemExecutionTimeReached(self, request, context):
		try:
			exec_time = request.value
			with self.fedexec_signals.lock:
				is_reached = self.fedexec_signals.is_execution_time_reached(exec_time)
			ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=is_reached)
			return ack_pb
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 2: {}".format(e))
			logging.error(traceback.format_exc())


	def IsLearnerCommunityUpdateSignalReached(self, request, context):
		try:
			learner_id = request.learner_id
			with self.fedround_spotter.lock:
				learner_completed_epochs = self.fedround_spotter.retrieve_fedround_learner_completed_epochs(learner_id)
			proceed_for_update = self.fedexec_signals.is_target_epochs_reached(learner_completed_epochs, synchronous_execution=self.synchronous_execution)
			ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=proceed_for_update)
			return ack_pb
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 3: {}".format(e))
			logging.error(traceback.format_exc())


	def RetrieveFederationRoundExecutionMetadataFromController(self, request, context):
		try:
			learner_id = request.learner_id
			with self.fedround_spotter.lock:
				fedround_spotter_json = self.fedround_spotter.toJSON_representation()
			fedround_spotter_json = json.dumps(fedround_spotter_json)
			jsonstring_pb = ProtoBuffExchangeOps.construct_json_string_value_pb(json_string=fedround_spotter_json)
			return jsonstring_pb
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 4: {}".format(e))
			logging.error(traceback.format_exc())


	def ResetControllerFederationRoundCollections(self, request, context):
		try:
			metis_logger.info('GRPC Controller. Federation Round {} ended. Collective execution signals: {}'.format(self.current_federation_round, self.fedround_spotter))
			with self.lock:
				self.fedround_learners_community_state = LearnersExecutionStateMap() # clear all learners state
				self.fedround_spotter = FedRoundSpotter(self.system_learners) # clear federation round spotter
				self.current_federation_round += 1 # proceed to the next round
			return ProtoBuffExchangeOps.construct_ack_pb(status=True)
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 5: {}".format(e))
			logging.error(traceback.format_exc())


	def NotifyControllerToUpdateFederationRoundSignals(self, request_iterator, context):
		""" Updating Controller execution signals, such as epoch completion"""

		try:
			for request in request_iterator:

				learner_id = request.learner.learner_id
				learner_finished_epoch = request.learner_finished_epoch
				learner_finished_training = request.learner_finished_training

				terminate = False
				if learner_finished_epoch:
					with self.fedround_spotter.lock:
						self.fedround_spotter.update_fedround_global_completed_epochs(value=1)
						self.fedround_spotter.update_fedround_learner_completed_epochs(learner_id=learner_id, value=1)
						# Following inspection applies only to semi-synchronous execution
						fedround_completed_epochs = self.fedround_spotter.retrieve_fedround_completed_epochs()
					terminate = self.fedexec_signals.is_target_epochs_reached(fedround_completed_epochs)
					if terminate and self.semi_synchronous_execution:
						metis_logger.info('GRPC Controller notifies learners that federation round signal {}-epochs reached.'.format(self.fedexec_signals.target_epochs))

				if learner_finished_training:
					with self.fedround_spotter.lock:
						self.fedround_spotter.update_fedround_global_completed_learners(value=1)
						self.fedround_spotter.update_fedround_learner_finished_training(learner_id=learner_id, value=True)
					# Following inspection applies only to semi-synchronous execution
					fedround_completed_learners = self.fedround_spotter.retrieve_fedround_completed_learners()
					terminate = self.fedexec_signals.is_target_learners_reached(fedround_completed_learners)
					if terminate and self.semi_synchronous_execution:
							metis_logger.info('GRPC Controller notifies learners that federation round signal {}-learners reached.'.format(self.fedexec_signals.target_learners))

				if terminate:
					# TODO The following condition is toggled only when the execution occurs in a semi-synchronous policy
					with self.fedround_spotter.lock:
						self.fedround_spotter.update_fedround_termination_signal_reached(True)
				return ProtoBuffExchangeOps.construct_ack_pb(status=True)

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 6: {}".format(e))
			logging.error(traceback.format_exc())


	def NotifyLearnersFederationRoundSignalReached(self, request, context):

		""" Function used for (semi)synchronous execution. Stop federation round if #target_epochs are reached, or #target_learners have finished. """

		try:
			while True:

				with self.fedround_spotter.lock:
					# Inform clients whether the federation round conditions have been met
					fedround_termination_reached = self.fedround_spotter.retrieve_fedround_termination_signal_reached()
				ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=fedround_termination_reached)
				yield ack_pb

				time.sleep(_LOOPBACK_SLEEP_SECONDS)

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 7: {}".format(e))
			logging.error(traceback.format_exc())


	def NotifyLearnersSessionTerminationSignalReached(self, request, context):

		""" Function used for primarily for asynchronous execution. If the total execution time is reached then notify learners to shut down. """

		with self.fedexec_signals.lock:
			max_sleep_time_mins = self.fedexec_signals.target_exec_time_mins
		current_min = 0
		one_minute_sleep = 60
		try:
			while True:
				# Periodic sleep for the requested execution minutes
				if current_min < max_sleep_time_mins:
					current_min += 1
					time.sleep(one_minute_sleep)
					# ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=False)
				else:
					metis_logger.info('Reached session termination time threshold.')
					ack_pb = ProtoBuffExchangeOps.construct_ack_pb(status=True)
					yield ack_pb
					time.sleep(_LOOPBACK_SLEEP_SECONDS)
					return
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 8: {}".format(e))
			logging.error(traceback.format_exc())


	def SendLearnerExecutionResultToController(self, request_iterator, context):

		""" Just insert requester's/learner's state in the community or update existing. """
		try:
			for learner_exec_msg in request_iterator:

				# self.community_update_requests_q.put(learner_exec_msg)
				learner_id = learner_exec_msg.execution_metadata.learner.learner_id
				learner_num_training_examples = learner_exec_msg.execution_metadata.num_training_examples
				learner_num_validation_examples = learner_exec_msg.execution_metadata.num_validation_examples
				learner_latest_train_score = learner_exec_msg.execution_metadata.latest_train_score
				learner_latest_validation_score = learner_exec_msg.execution_metadata.latest_validation_score
				learner_completed_epochs = learner_exec_msg.execution_metadata.completed_epochs
				learner_completed_batches = learner_exec_msg.execution_metadata.completed_batches
				learner_batch_size = learner_exec_msg.execution_metadata.batch_size
				learner_processing_ms_per_epoch = learner_exec_msg.execution_metadata.processing_ms_per_epoch
				learner_target_local_epochs = learner_exec_msg.execution_metadata.target_local_epochs
				learner_network_matrices_pb = learner_exec_msg.network_matrices.matrices

				# We construct ndarrays from the protobuff network_matrices
				nd_arrays = ProtoBuffExchangeOps.reconstruct_ndarrays_from_network_matrices_pb(network_matrices_pb=learner_network_matrices_pb)

				# Check if learner has already issued requests. If yes, update its update counter request by 1, if no then assign 1
				learner_current_update_num = 1
				prev_learner_global_update_num = 0
				prev_learner_community_update_steps_num = 0
				prev_learner_completed_batches = 0

				with self.fedround_learners_community_state.lock:

					if self.fedround_learners_community_state.learner_exists(learner_id):
						prev_learner_update_num = self.fedround_learners_community_state.retrieve_learner_update_request_num(learner_id)
						learner_current_update_num = prev_learner_update_num + 1
						prev_learner_global_update_num = self.fedround_learners_community_state.retrieve_learner_global_update_num(learner_id)
						prev_learner_community_update_steps_num = self.fedround_learners_community_state.retrieve_learner_current_community_global_steps_num(learner_id)
						prev_learner_completed_batches = self.fedround_learners_community_state.retrieve_learner_previous_completed_batches(learner_id)

					# Increase global update request counter and assign it to the learner
					self.fedround_learners_community_state.increase_global_request_num_by1()
					learner_current_global_update_num = self.fedround_learners_community_state.global_update_request_num

					current_community_update_steps = self.fedround_learners_community_state.retrieve_current_community_global_steps()
					learner_current_update_steps = learner_completed_batches - prev_learner_completed_batches
					self.fedround_learners_community_state.increase_global_steps_by_val(val=learner_current_update_steps)

					learner_exec_res = LearnerExecutionResultWihMatrices(learner_id=learner_id, network_matrices=nd_arrays, num_training_examples=learner_num_training_examples, num_validation_examples=learner_num_validation_examples,
																		 latest_train_score=learner_latest_train_score, latest_validation_score=learner_latest_validation_score, completed_epochs=learner_completed_epochs,
																		 completed_batches=learner_completed_batches, batch_size=learner_batch_size, processing_ms_per_epoch=learner_processing_ms_per_epoch, update_request_num=learner_current_update_num,
																		 global_update_num=learner_current_global_update_num, previous_global_update_num=prev_learner_global_update_num, target_local_epochs=learner_target_local_epochs,
																		 previous_completed_batches=prev_learner_completed_batches, previous_community_update_steps=prev_learner_community_update_steps_num, current_community_update_steps=current_community_update_steps)

					# Update Controller's Learners state map
					self.fedround_learners_community_state.update_state_map(learner_exec_res)

				ack = ProtoBuffExchangeOps.construct_ack_pb(status=True)
				return ack

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 9: {}".format(e))
			logging.error(traceback.format_exc())


	def CommunityUpdateWithLearnerState(self, request_iterator, context):

		""" Request a Community update including the requester's/learner's state. State map is updated during computation. """

		try:

			for learner_exec_msg in request_iterator:

				# self.community_update_requests_q.put(learner_exec_msg)
				learner_id = learner_exec_msg.execution_metadata.learner.learner_id
				metis_logger.info("Handling community update request of learner: {}".format(learner_id))

				learner_num_training_examples = learner_exec_msg.execution_metadata.num_training_examples
				learner_num_validation_examples = learner_exec_msg.execution_metadata.num_validation_examples
				learner_latest_train_score = learner_exec_msg.execution_metadata.latest_train_score
				learner_latest_validation_score = learner_exec_msg.execution_metadata.latest_validation_score
				learner_completed_epochs = learner_exec_msg.execution_metadata.completed_epochs
				learner_completed_batches = learner_exec_msg.execution_metadata.completed_batches
				learner_batch_size = learner_exec_msg.execution_metadata.batch_size
				learner_processing_ms_per_epoch = learner_exec_msg.execution_metadata.processing_ms_per_epoch
				learner_target_local_epochs = learner_exec_msg.execution_metadata.target_local_epochs
				learner_network_matrices_pb = learner_exec_msg.network_matrices.matrices

				# We construct ndarrays from the protobuff network_matrices
				nd_arrays = ProtoBuffExchangeOps.reconstruct_ndarrays_from_network_matrices_pb(network_matrices_pb=learner_network_matrices_pb)

				# Check if learner has already issued requests. If yes, update its update counter request by 1, if no then assign 1
				learner_current_update_num = 1
				prev_learner_global_update_num = 0
				prev_learner_community_update_steps_num = 0
				prev_learner_completed_batches = 0

				with self.fedround_learners_community_state.lock:
					if self.fedround_learners_community_state.learner_exists(learner_id):
						prev_learner_update_num = self.fedround_learners_community_state.retrieve_learner_update_request_num(learner_id)
						learner_current_update_num = prev_learner_update_num + 1
						prev_learner_global_update_num = self.fedround_learners_community_state.retrieve_learner_global_update_num(learner_id)
						prev_learner_community_update_steps_num = self.fedround_learners_community_state.retrieve_learner_current_community_global_steps_num(learner_id)
						prev_learner_completed_batches = self.fedround_learners_community_state.retrieve_learner_previous_completed_batches(learner_id)

					# Increase global update request counter and assign it to the learner
					self.fedround_learners_community_state.increase_global_request_num_by1()
					learner_current_global_update_num = self.fedround_learners_community_state.global_update_request_num

					current_community_update_steps = self.fedround_learners_community_state.retrieve_current_community_global_steps()
					learner_current_update_steps = learner_completed_batches - prev_learner_completed_batches
					self.fedround_learners_community_state.increase_global_steps_by_val(val=learner_current_update_steps)

					learner_exec_res = LearnerExecutionResultWihMatrices(learner_id=learner_id, network_matrices=nd_arrays, num_training_examples=learner_num_training_examples, num_validation_examples=learner_num_validation_examples,
																		 latest_train_score=learner_latest_train_score, latest_validation_score=learner_latest_validation_score, completed_epochs=learner_completed_epochs,
																		 completed_batches=learner_completed_batches, batch_size=learner_batch_size, processing_ms_per_epoch=learner_processing_ms_per_epoch, update_request_num=learner_current_update_num,
																		 global_update_num=learner_current_global_update_num, previous_global_update_num=prev_learner_global_update_num, target_local_epochs=learner_target_local_epochs,
																		 previous_completed_batches=prev_learner_completed_batches, previous_community_update_steps=prev_learner_community_update_steps_num, current_community_update_steps=current_community_update_steps)

					if self.fedround_learners_community_state.number_of_cached_learners() >= self.minimum_learners_for_community_update:
						# The following is the suggested optimization, use of moving average and cache
						community_network_matrices = self.fedround_learners_community_state.compute_community_network(learner_exec_res, using_cache=True)
						# Convert community ndarrays to protobuff ops
						community_network_matrices_pb = ProtoBuffExchangeOps.construct_network_matrices_pb_from_ndarrays(ndarrays=community_network_matrices)
					else:
						# Do not perform any community computation. Simply update the federation state map
						self.fedround_learners_community_state.update_state_map(learner_exec_res)
						# Since the required number of learners has not been reached yet, we simply return the arrays back to the learner as they are
						community_network_matrices_pb = learner_exec_msg.network_matrices

				return community_network_matrices_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 10: {}".format(e))
			logging.error(traceback.format_exc())


	def CommunityUpdateCurrentState(self, request, context):

		""" Just return the current community state. """

		try:
			learner_id = request.learner_id
			# In synchronous mode compute community from scratch, is asynchronous mode compute community through the cached states
			with self.fedround_learners_community_state.lock:
				if self.synchronous_execution:
					community_network_matrices = self.fedround_learners_community_state.compute_community_network(using_cache=False)
				else:
					community_network_matrices = self.fedround_learners_community_state.compute_community_network(using_cache=True)
			network_matrices_pb = ProtoBuffExchangeOps.construct_network_matrices_pb_from_ndarrays(ndarrays=community_network_matrices)
			return network_matrices_pb
		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 11: {}".format(e))
			logging.error(traceback.format_exc())


	def LearnerLocalIterationsRequest(self, request, context):
		try:
			learner_id = request.learner.learner_id
			with self.fedround_spotter.lock:
				assigned_target_local_epochs = self.fedround_spotter.compute_learner_target_local_iterations(learner_id)
			integer_value_pb = ProtoBuffExchangeOps.construct_integer_value_pb(val=assigned_target_local_epochs)
			return integer_value_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 13: {}".format(e))
			logging.error(traceback.format_exc())


	def LearnerCommunityStateMetadata(self, request, context):
		try:

			with self.fedround_learners_community_state.lock:
				learner_id = request.learner_id
				learner_global_update_scalar_clock = 1
				global_update_scalar_clock = self.fedround_learners_community_state.retrieve_current_global_update_request_num()

				learner_previous_community_steps = 1
				current_community_global_steps = self.fedround_learners_community_state.retrieve_current_community_global_steps()

				if self.fedround_learners_community_state.learner_exists(learner_id):
					learner_global_update_scalar_clock = self.fedround_learners_community_state.retrieve_learner_global_update_num(learner_id)
					learner_previous_community_steps = self.fedround_learners_community_state.retrieve_learner_previous_community_global_steps_num(learner_id)

			community_state_pb = ProtoBuffExchangeOps.construct_community_state_metadata(global_update_scalar_clock=global_update_scalar_clock,
																						 learner_global_update_scalar_clock=learner_global_update_scalar_clock,
																						 global_community_steps=current_community_global_steps,
																						 learner_previous_global_community_steps=learner_previous_community_steps)
			return community_state_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 14: {}".format(e))
			logging.error(traceback.format_exc())


	def LearnerStalenessCommunityRequestEligibility(self, request, context):
		try:
			learner_id = request.learner.learner_id
			learner_validation_phase_stalenesses = request.validation_phase_stalenesses
			learner_current_update_steps = request.current_update_steps

			with self.fedround_learners_community_state.lock:

				# TODO THIS IS TO ALLEVIATE THE LARGE DROP ON THE COMMUNITY MODEL DURING TRANSITION
				gid = self.fedround_learners_community_state.retrieve_current_global_epoch(
					num_system_learners=self.num_system_learners)
				if gid > 100 and self.__transition_community_computation is False:
					self.fedround_learners_community_state.reset_community_state_network_map()
					self.__transition_community_computation = True


				self.fedround_learners_community_state.validation_phase_staleness_map[learner_id] = \
					learner_validation_phase_stalenesses

				community_eligibility = False
				if len(self.fedround_learners_community_state.validation_phase_staleness_map.keys()) == self.num_system_learners and \
					self.fedround_learners_community_state.learner_exists(learner_id):
						# Need to flatten this. Otherwise we raise exceptions.
						historical_stalenesses = [x for v in self.fedround_learners_community_state.validation_phase_staleness_map.values()
												  for x in v]
						historical_staleness_mean = np.mean(historical_stalenesses)
						historical_staleness_median = np.median(historical_stalenesses)

						current_learners_staleness = self.fedround_learners_community_state.get_current_community_update_steps_staleness(
							new_update_steps=learner_current_update_steps)
						current_learners_staleness_mean = np.mean(list(current_learners_staleness.values()))
						current_learners_staleness_median = np.median(list(current_learners_staleness.values()))
						metis_logger.info(msg="{} Validation Phases Staleness Value: {}, Current Community Staleness Value: {}".format(
							learner_id, historical_staleness_median, current_learners_staleness_median))

						if current_learners_staleness_median >= historical_staleness_median:
							community_eligibility = True

			community_eligibility_pb = ProtoBuffExchangeOps.construct_ack_pb(status=community_eligibility)
			return community_eligibility_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 15: {}".format(e))
			logging.error(traceback.format_exc())


	def RequestGlobalEpochID(self, request, context):
		try:
			global_epoch = self.fedround_learners_community_state.retrieve_current_global_epoch(
				num_system_learners=self.num_system_learners)
			global_epoch_pb = ProtoBuffExchangeOps.construct_integer_value_pb(global_epoch)
			return global_epoch_pb

		except Exception as e:
			metis_logger.info("CONTROLLER ERROR 16: {}".format(e))
			logging.error(traceback.format_exc())


class GRPCController(GRPCServer):

	def __init__(self, grpc_servicer, executor):
		GRPCServer.__init__(self, grpc_servicer, thread_pool_executor=executor)

	def start(self):
		collaborative_learning_pb2_grpc.add_CollabLearningServicer_to_server(
			servicer=self.grpc_servicer,
			server=self.server)
		self.server.add_insecure_port(self.grpc_servicer.host_port)
		try:
			self.server.start()
		except Exception as e:
			metis_logger.info(msg="Metis grpc controller threw an exception: {}".format(e))
		try:
			while True:
				time.sleep(self.service_lifetime)
				metis_logger.info("Controller is still running")
		except KeyboardInterrupt:
			# TODO change this if needed
			self.server.stop(grace=200) # grace timeout of 10secs, default was 0 (None)

	def stop(self):
		stopping_event = self.server.stop(grace=0) # grace timeout of 0 (None)
		return stopping_event

class FedController(object):
	"""
	A helper class for initializing Federation Controller
	"""

	def __init__(self, grpc_servicer_host_port, participating_hosts_ids, synchronous_execution=True, target_learners=0, target_epochs=0, target_score=0, target_exec_time_mins=0,
				 required_learners_for_community_update=1, max_workers=100):
		self.__executor = ThreadPoolExecutor(max_workers=max_workers,
											 thread_name_prefix='PoolExecutorOf_{}'.format("GRPCController"))
		self.__controller_grpc_servicer = SystemControllerServicer(host_port=grpc_servicer_host_port,
																   synchronous_execution=synchronous_execution,
																   system_learners=participating_hosts_ids,
																   target_learners=target_learners,
																   target_epochs=target_epochs,
																   target_score=target_score,
																   target_exec_time_mins=target_exec_time_mins,
																   minimum_learners_for_community_update=required_learners_for_community_update)

		'''
		The max workers is a required property. It specifies the number of concurrent threads handling requests on the server.
		Background requests from the clients can saturate server's active threads.
		In the current implementation, every grpc client will have at least one active background listener for incoming messages. 
		One listener is for federation round termination signals (semi-synchronous execution) and another for session termination signals.
		Overall, if all clients are alive then the total number of workers should be at least: 
			max_workers = #clients * 2
		'''
		self.__controller_grpc_server = GRPCController(self.__controller_grpc_servicer, self.__executor)
		# self.__server_process = None

		self.grpc_server_future = None

	def start(self):
		metis_logger.info(msg='Initializing GRPC Metis Controller.')
		# self.__server_process = multiprocessing.Process(target=self.__controller_grpc_server.start, name="FederationController")
		# self.__server_process.daemon = True
		# self.__server_process.start()
		self.grpc_server_future = self.__executor.submit(self.__controller_grpc_server.start)
		metis_logger.info(msg='GRPC Metis Controller Initialized @ {}'.format(self.__controller_grpc_servicer.host_port))
		time.sleep(0.1) # Wait till the Parameter Server is fully initialized

	def stop(self):
		metis_logger.info(msg='Shutting Down GRPC Metis Controller.')
		self.__controller_grpc_server.stop()
		# sig = signal.SIGTERM
		# os.kill(self.__server_process.pid, sig)

		# TODO Following this is a hack in order to signal that the thread pool executor needs to shut down.
		#  Check solution at stackoverflow:
		#  https://stackoverflow.com/questions/48350257/how-to-exit-a-script-after-threadpoolexecutor-has-timed-out
		try:
			self.grpc_server_future.result(timeout=1)
		except TimeoutError as time_error:
			import atexit
			atexit.unregister(concurrent.futures.thread._python_exit)
			self.__executor.stop = lambda wait: None
			self.__executor.shutdown(wait=False)
			self.__controller_grpc_server.executor.shutdown(wait=False)
		metis_logger.info(msg='GRPC Metis Controller shut down.')