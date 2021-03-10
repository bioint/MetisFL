import json

import numpy as np

from federation.fed_protobuff import collaborative_learning_pb2
from federation.fed_protobuff import model_evaluation_serving_pb2


class ProtoBuffExchangeOps(object):


	@classmethod
	def construct_network_matrices_pb_from_ndarrays(cls, ndarrays):
		if not any(isinstance(ndarray, np.ndarray) for ndarray in ndarrays):
			raise TypeError("Parameter `ndarrays` must be of type %s " % np.ndarray)
		network_matrices = list()
		for ndarray in ndarrays:
			flatten_var = ndarray.flatten()
			var_size = int(ndarray.size)
			var_dtype = str(ndarray.dtype.name)
			var_shape = ndarray.shape
			matrix_def_pb = collaborative_learning_pb2.MatrixDef(size=var_size, dtype=var_dtype, dimensions=var_shape)
			double_matrix_pb = collaborative_learning_pb2.DoubleMatrix(values=flatten_var, matrixdef=matrix_def_pb)
			network_matrices.append(double_matrix_pb)
		network_matrices_pb = collaborative_learning_pb2.NetworkMatrices(matrices=network_matrices)
		return network_matrices_pb


	@classmethod
	def reconstruct_ndarrays_from_network_matrices_pb(cls, network_matrices_pb):
		nd_arrays = list()
		for matrix in network_matrices_pb:
			matrix_def = matrix.matrixdef
			matrix_values = matrix.values
			matrix_size = matrix_def.size
			matrix_dtype = matrix_def.dtype
			matrix_dimensions = matrix_def.dimensions
			ndarray = np.array(matrix_values, dtype=matrix_dtype)
			ndarray = np.reshape(ndarray, matrix_dimensions)
			nd_arrays.append(ndarray)
		return nd_arrays


	@classmethod
	def construct_empty_message_pb(cls):
		return collaborative_learning_pb2.EmptyMessage()


	@classmethod
	def construct_double_value_pb(cls, val):
		return collaborative_learning_pb2.DoubleValue(value=val)


	@classmethod
	def construct_integer_value_pb(cls, val):
		return collaborative_learning_pb2.IntegerValue(value=val)


	@classmethod
	def construct_json_string_value_pb(cls, json_string):
		try:
			json.loads(json_string)
		except ValueError:
			print("%s is not a valid json" % json_string)
			return collaborative_learning_pb2.JsonStringValue()
		finally:
			return collaborative_learning_pb2.JsonStringValue(value=json_string)


	@classmethod
	def construct_learner_pb(cls, learner_id):
		return collaborative_learning_pb2.Learner(learner_id=learner_id)


	@classmethod
	def construct_ack_pb(cls, status):
		return collaborative_learning_pb2.Ack(status=status)


	@classmethod
	def contstruct_fedround_signal(cls, learner_id, finished_epoch, finished_training):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=learner_id)
		return collaborative_learning_pb2.FedRoundSignals(learner=learner_pb,
														  learner_finished_epoch=finished_epoch,
														  learner_finished_training=finished_training)


	@classmethod
	def construct_empty_network_matrix(cls):
		return collaborative_learning_pb2.NetworkMatrices()


	@classmethod
	def construct_learner_execution_result(cls, learner_id, matrices, num_training_examples, num_validation_examples,
										   latest_train_score, latest_validation_score, comp_epochs, comp_batches,
										   batch_size, processing_ms_per_epoch, target_local_epochs):
		network_matrices_pb = ProtoBuffExchangeOps.construct_network_matrices_pb_from_ndarrays(ndarrays=matrices)
		execution_result_metadata_pb = ProtoBuffExchangeOps.construct_learner_execution_metadata(
			learner_id=learner_id, num_training_examples=num_training_examples,
			num_validation_examples=num_validation_examples, latest_train_score=latest_train_score,
			latest_validation_score=latest_validation_score, comp_epochs=comp_epochs, comp_batches=comp_batches,
			batch_size=batch_size, processing_ms_per_epoch=processing_ms_per_epoch,
			target_local_epochs=target_local_epochs)
		return collaborative_learning_pb2.LearnerExecutionResult(network_matrices=network_matrices_pb,
																 execution_metadata=execution_result_metadata_pb)


	@classmethod
	def construct_model_evaluation_request(cls, learner_id, matrices, num_training_examples, num_validation_examples,
										   latest_train_score, latest_validation_score, comp_epochs, comp_batches,
										   batch_size, processing_ms_per_epoch, target_local_epochs,
										   is_community_model):
		network_matrices_pb = ProtoBuffExchangeOps.construct_network_matrices_pb_from_ndarrays(ndarrays=matrices)
		execution_result_metadata_pb = ProtoBuffExchangeOps.construct_learner_execution_metadata(
			learner_id=learner_id, num_training_examples=num_training_examples,
			num_validation_examples=num_validation_examples, latest_train_score=latest_train_score,
			latest_validation_score=latest_validation_score, comp_epochs=comp_epochs, comp_batches=comp_batches,
			batch_size=batch_size, processing_ms_per_epoch=processing_ms_per_epoch,
			target_local_epochs=target_local_epochs)
		learner_execution_result_pb = collaborative_learning_pb2.LearnerExecutionResult(
			network_matrices=network_matrices_pb, execution_metadata=execution_result_metadata_pb)
		model_evaluation_request_pb = model_evaluation_serving_pb2.ModelEvaluationRequest(
			is_community_model=is_community_model, learner_execution_result=learner_execution_result_pb)
		return model_evaluation_request_pb


	@classmethod
	def construct_learner_execution_metadata(cls, learner_id, num_training_examples, num_validation_examples,
											 latest_train_score, latest_validation_score, comp_epochs, comp_batches,
											 batch_size, processing_ms_per_epoch, target_local_epochs):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=learner_id)
		return collaborative_learning_pb2.LearnerExecutionMetadata(learner=learner_pb,
																   num_training_examples=num_training_examples,
																   num_validation_examples=num_validation_examples,
																   latest_train_score=latest_train_score,
																   latest_validation_score=latest_validation_score,
																   completed_epochs=comp_epochs,
																   completed_batches=comp_batches,
																   batch_size=batch_size,
																   processing_ms_per_epoch=processing_ms_per_epoch,
																   target_local_epochs=target_local_epochs)


	@classmethod
	def construct_community_state_metadata(cls, global_update_scalar_clock, learner_global_update_scalar_clock,
										   global_community_steps, learner_previous_global_community_steps):
		return collaborative_learning_pb2.CommunityStateMetadata(
			global_update_scalar_clock=global_update_scalar_clock,
			learner_global_update_scalar_clock=learner_global_update_scalar_clock,
			global_community_steps=global_community_steps,
			learner_previous_global_community_steps=learner_previous_global_community_steps)


	@classmethod
	def construct_learner_staleness_metadata(cls, learner_id, validation_phase_stalenesses, current_update_steps):
		learner_pb = ProtoBuffExchangeOps.construct_learner_pb(learner_id=learner_id)
		return collaborative_learning_pb2.LearnerStalenessMetadata(
			learner=learner_pb, validation_phase_stalenesses=validation_phase_stalenesses,
			current_update_steps=current_update_steps)
