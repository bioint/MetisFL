import concurrent
import federation.fed_model as fedmodel
import json
import threading
import time

import federation.fed_cluster_env as fedenv
import numpy as np
import tensorflow as tf

from collections import defaultdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from federation.fed_protobuff import model_evaluation_serving_pb2_grpc
from metisdb.metisdb_session import MetisDBSession
from utils.devops.network_ops import NetOpsUtil
from utils.devops.grpc_services import GRPCServer
from utils.devops.proto_buff_exchange_ops import ProtoBuffExchangeOps
from utils.generic.time_ops import TimeUtil
from utils.logging.metis_logger import MetisLogger as metis_logger

from utils.tf.tf_ops_configs import TFConfiguration
from utils.tf.tf_ops_evaluation import TFGraphEvaluation

tf.logging.set_verbosity(tf.logging.DEBUG)  # Enable INFO verbosity


class TFDVWEvaluationGraphOps(object):

	def __init__(self, eval_graph, tf_eval_ops, variables_init_op, assign_atomic_fed_placeholders,
				 assign_atomic_feds_ops, x_placeholders, y_placeholders, hosts_validation_dataset_ops):
		self.eval_graph = eval_graph
		self.tf_eval_ops = tf_eval_ops
		self.variables_init_op = variables_init_op
		self.assign_atomic_fed_placeholders = assign_atomic_fed_placeholders
		self.assign_atomic_feds_ops = assign_atomic_feds_ops
		self.x_placeholders = x_placeholders
		self.y_placeholders = y_placeholders
		self.hosts_validation_dataset_ops = hosts_validation_dataset_ops


class SingleValidationResult(object):

	def __init__(self, learner_id, validation_dataset_size, validation_batches_num, validation_eval_metrics,
				 validation_score_name, validation_score, validation_loss_sum, validation_loss_squared_sum,
				 validation_loss_mean, validation_loss_variance):
		self.learner_id = learner_id
		self.validation_dataset_size = validation_dataset_size
		self.validation_batches_num = validation_batches_num
		self.validation_eval_metrics = validation_eval_metrics
		self.validation_score_name = validation_score_name
		self.validation_score = validation_score
		self.validation_loss_sum = validation_loss_sum
		self.validation_loss_squared_sum = validation_loss_squared_sum
		self.validation_loss_mean = validation_loss_mean
		self.validation_loss_variance = validation_loss_variance

	def __repr__(self):
		return "({}, {}, {})".format(self.learner_id, self.validation_dataset_size, self.validation_score)

	def toJSON_representation(self):
		return {"learner_id": self.learner_id,
				"validation_dataset_size": int(self.validation_dataset_size),
				"validation_batches_num": int(self.validation_batches_num),
				self.validation_score_name: float(self.validation_score),
				"validation_loss_sum": float(self.validation_loss_sum),
				"validation_loss_squared_sum": float(self.validation_loss_squared_sum),
				"validation_loss_mean": float(self.validation_loss_mean),
				"validation_loss_variance": float(self.validation_loss_variance)}


class EvaluationRequestMeta(object):

	def __init__(self, learner_id, request_init_unix_time_ms, request_end_unix_time_ms, is_community_model,
				 learner_validation_weight, learner_federated_validation):
		self.learner_id = learner_id
		self.request_init_unix_time_ms = request_init_unix_time_ms
		self.request_end_unix_time_ms = request_end_unix_time_ms
		self.is_community_model = is_community_model
		self.learner_validation_weight = learner_validation_weight
		self.learner_federated_validation = learner_federated_validation

	def toJSON_representation(self):
		if self.learner_federated_validation is not None:
			json_learner_federated_validation = [res.toJSON_representation() for res in
												 self.learner_federated_validation]
		else:
			json_learner_federated_validation = []
		return {"learner_id": self.learner_id,
				"request_init_unix_time": self.request_init_unix_time_ms,
				"request_end_unix_time": self.request_end_unix_time_ms,
				"is_community_model": self.is_community_model,
				"learner_validation_weight": self.learner_validation_weight,
				"learner_federated_validation": json_learner_federated_validation}


class ModelEvaluatorServicer(model_evaluation_serving_pb2_grpc.EvalServingServicer):

	def __init__(self, fed_environment, federated_variables, federation_model_obj, metis_db_session, target_stat_name,
				 executor):

		if not isinstance(fed_environment, fedenv.FedEnvironment):
			raise TypeError("`fed_environment` must be of type %s " % fedenv.FedEnvironment)
		if not any(isinstance(fedvar, fedmodel.FedVar) for fedvar in federated_variables):
			raise TypeError("All the federated variables passed to this function must be of type %s " % fedmodel.FedVar)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s " % fedmodel.FedModelDef)
		if not isinstance(metis_db_session, MetisDBSession):
			raise TypeError("The `metis_db_session` parameter must be of type %s " % MetisDBSession)

		self.fed_environment = fed_environment
		self.community_function = self.fed_environment.community_function
		self.grpc_servicer_host_port = self.fed_environment.federation_evaluator_grpc_servicer_endpoint
		self.grpc_servicer_host = self.grpc_servicer_host_port.split(":")[0]
		self.grpc_servicer_port = self.grpc_servicer_host_port.split(":")[1]

		self.tf_ps_server_host_port = self.fed_environment.federation_evaluator_tensorflow_ps_endpoint
		self.tf_ps_server_host = self.tf_ps_server_host_port.split(":")[0]
		self.tf_ps_server_port = self.tf_ps_server_host_port.split(":")[1]

		# Check whether Tensorflow Evaluation Server is initialized.
		is_host_listening = NetOpsUtil.is_endpoint_listening(host=self.tf_ps_server_host, port=self.tf_ps_server_port)
		if not is_host_listening:
			raise RuntimeError("Endpoint %s is down. Resurrect it from the dead." % self.tf_ps_server_host_port)

		self.federated_variables = federated_variables
		self.federation_model_obj = federation_model_obj
		self.metis_db_session = metis_db_session
		self.learners_ids = self.metis_db_session.get_federation_learners_ids()
		self.target_stat_name = target_stat_name
		self.batch_size = self.fed_environment.fed_evaluator_tf_host.fed_worker_servers[0].batch_size
		self.evaluation_requests_meta = list()
		self.__model_eval_lock = threading.Lock()
		self.__thread_executor = executor
		self.__learners_validation_requests_loss_state_map = defaultdict(list)
		self.__community_model_validation_requests_loss = list()
		self.__community_model_validation_requests_futures = list()

		# Tensorflow graph and session specific configurations.
		self.session_config = TFConfiguration.tf_session_config(per_process_gpu_memory_fraction=0.5)
		evaluation_graph_tf_worker_server_endpoint = "grpc://{}:{}".format(self.tf_ps_server_host,
																		   int(self.tf_ps_server_port) + 1)
		evaluation_graph_tf_server_cluster = {
			'ps': ["{}:{}".format(self.tf_ps_server_host, self.tf_ps_server_port)],
			'worker': ["{}:{}".format(self.tf_ps_server_host, int(self.tf_ps_server_port) + 1)],
		}
		evaluation_graph_server_cluster_spec = tf.train.ClusterSpec(
			cluster=evaluation_graph_tf_server_cluster
		)
		self.__evaluation_graph_server_device_fn = tf.train.replica_device_setter(
			cluster=evaluation_graph_server_cluster_spec
		)
		self.__dvw_evaluation_graph_ops = self.__build_dvw_evaluation_graph_ops()
		with tf.device(self.__evaluation_graph_server_device_fn):
			with self.__dvw_evaluation_graph_ops.eval_graph.as_default() as g:
				# We create two separate evaluation graph sessions. One for local models and one for community models.
				# By creating two graph sessions we are able to scale better the evaluation of incoming models.
				self.local_models_evaluation_graph_session = tf.train.MonitoredTrainingSession(
					master=evaluation_graph_tf_worker_server_endpoint,
					is_chief=True,
					config=self.session_config)
				self.local_models_evaluation_graph_session.run(self.__dvw_evaluation_graph_ops.variables_init_op)
				self.community_models_evaluation_graph_session = tf.train.MonitoredTrainingSession(
					master=evaluation_graph_tf_worker_server_endpoint,
					is_chief=True,
					config=self.session_config)
				self.community_models_evaluation_graph_session.run(self.__dvw_evaluation_graph_ops.variables_init_op)

		# Delta Weighting
		self.delta_weighting_state_map = dict()


	def __build_dvw_evaluation_graph_ops(self):

		with tf.device(self.__evaluation_graph_server_device_fn):
			tf.reset_default_graph()
			evaluation_graph = tf.Graph()
			with evaluation_graph.as_default():

				_x_placeholders = self.federation_model_obj.input_tensors_datatype()
				_y_placeholders = self.federation_model_obj.output_tensors_datatype()

				# Define Deep NN Graph
				global_step = tf.train.get_or_create_global_step()
				model_architecture = self.federation_model_obj.model_architecture(
					input_tensors=_x_placeholders, output_tensors=_y_placeholders,
					model_variables=self.federated_variables, global_step=global_step, batch_size=None,
					dataset_size=None)

				# Register federated variables collection in the tf evaluation graph.
				with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
					for variable in self.federated_variables:
						tf.get_variable(name=variable.name, initializer=variable.value,
										collections=variable.tf_collection, trainable=variable.trainable)

				# Get trainable variables collection defined in the model graph
				trainable_variables = model_architecture.model_federated_variables

				# Define operations for model federated variables initialization.
				fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
				assign_atomic_fed_phs = []
				assign_atomic_feds_ops = []
				for (index, model_var) in enumerate(trainable_variables):
					fed_var_ph = tf.placeholder(dtype=fed_vars[index].dtype)
					assign_atomic_fed_phs.append(fed_var_ph)
					assign_atomic_feds_ops.append(model_var.assign(fed_var_ph))

				tf_graph_evals = TFGraphEvaluation(_x_placeholders, _y_placeholders,
												   predictions_tensor_fedmodel=model_architecture.predictions,
												   loss_tensor_fedmodel=model_architecture.loss,
												   is_classification=self.metis_db_session.is_classification,
												   is_regression=self.metis_db_session.is_regression,
												   num_classes=self.metis_db_session.num_classes,
												   negative_classes_indices=self.metis_db_session.negative_classes_indices,
												   is_eval_output_scalar=self.metis_db_session.is_eval_output_scalar)

				# TODO We get the evaluation attribute of the first learner. Not necessarily a stable solution.
				tf_graph_evals.register_evaluation_ops(
					self.metis_db_session.get_learner_evaluation_output_attribute(self.learners_ids[0]), self.batch_size)

				# Import the validation (federation) dataset from all federation learners.
				hosts_valid_dataset_ops = dict()
				for learner_id in self.learners_ids:
					_, validation_dataset_structure, _ = self.metis_db_session.import_host_data(learner_id,
																								self.batch_size,
																								import_validation=True)
					hosts_valid_dataset_ops[learner_id] = validation_dataset_structure

				variables_init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

				# Define the TFDVW evaluation operations object.
				evaluation_graph_ops = TFDVWEvaluationGraphOps(eval_graph=evaluation_graph,
															   tf_eval_ops=tf_graph_evals,
															   variables_init_op=variables_init_op,
															   assign_atomic_fed_placeholders=assign_atomic_fed_phs,
															   assign_atomic_feds_ops=assign_atomic_feds_ops,
															   x_placeholders=_x_placeholders,
															   y_placeholders=_y_placeholders,
															   hosts_validation_dataset_ops=hosts_valid_dataset_ops)

			return evaluation_graph_ops


	def __compute_learner_weighting_value(self, learner_id, federation_evaluations, is_community_model):

		# Federation validation set total number of batches.
		federation_population_size = sum([fedeval.validation_batches_num for fedeval in federation_evaluations])

		# Federation validation set total/sum loss.
		federation_loss_total_sum = np.sum([fedeval.validation_loss_sum for fedeval in federation_evaluations])

		# Federation validation set batch-weighted sum loss.
		federation_loss_weighted_sum = np.sum([fedeval.validation_batches_num * fedeval.validation_loss_sum
											   for fedeval in federation_evaluations])
		federation_loss_weighted_sum /= federation_population_size

		# Federation validation set total/sum squared loss.
		sum_squared_federation_loss = np.sum(
			[fedeval.validation_loss_squared_sum for fedeval in federation_evaluations])

		# Federation validation set batch-weighted mean loss.
		federation_loss_weighted_mean = np.sum([fedeval.validation_batches_num * fedeval.validation_loss_mean
												for fedeval in federation_evaluations])
		federation_loss_weighted_mean = federation_loss_weighted_mean / federation_population_size

		# Federation validation set pooled loss variance.
		federation_loss_pooled_variance = np.sum(
			[(fedeval.validation_batches_num * fedeval.validation_loss_variance) +
			 (fedeval.validation_batches_num * np.power(fedeval.validation_loss_mean, 2))
			 for fedeval in federation_evaluations])
		federation_loss_pooled_variance = federation_loss_pooled_variance - federation_population_size * np.power(
			federation_loss_weighted_mean, 2)
		federation_loss_pooled_variance = federation_loss_pooled_variance / federation_population_size
		ifvl = np.divide(1, federation_loss_pooled_variance + sum_squared_federation_loss)
		metis_logger.info(msg="{}, Inverted Federation Validation Loss: {}".format(learner_id, str(ifvl)))

		if self.metis_db_session.is_classification:
			# Federation validation set batch-weighted sum accuracy.
			federation_accuracy_weighted_sum = np.sum([fedeval.validation_batches_num * fedeval.validation_score
												   for fedeval in federation_evaluations])
			federation_accuracy_weighted_sum /= federation_population_size

			collective_class_statistics = defaultdict(dict)
			collective_all_classes_statistics = Counter()
			for fedeval in federation_evaluations:
				evaluation_stats_per_class = fedeval.validation_eval_metrics['evaluation_per_class']
				# We use the counter to add the values of the dictionaries at once instead of looping over each one.
				for class_id in evaluation_stats_per_class:
					collective_class_statistics[class_id] = Counter(collective_class_statistics[class_id]) \
															+ Counter(evaluation_stats_per_class[class_id])
					collective_all_classes_statistics = collective_all_classes_statistics \
														+ Counter(evaluation_stats_per_class[class_id])

			# Micro average accuracy.
			micro_average_accuracy = np.divide(collective_all_classes_statistics['TP'],
											   collective_all_classes_statistics['NoExamples'])
			metis_logger.info(msg="{}, Micro-Average Accuracy: {}".format(learner_id, str(micro_average_accuracy)))

			# Cumulative statistics
			cumulative_tps = 0
			cumulative_fps = 0
			cumulative_fns = 0

			per_class_scores = list()
			for class_id in collective_class_statistics:

				if 'NoExamples' in collective_class_statistics[class_id]:
					num_examples = collective_class_statistics[class_id]['NoExamples']
				else:
					num_examples = 0

				if collective_class_statistics[class_id]:
					tps = collective_class_statistics[class_id]['TP']
					fps = collective_class_statistics[class_id]['FP']
					fns = collective_class_statistics[class_id]['FN']

					cumulative_tps += tps
					cumulative_fps += fps
					cumulative_fns += fns

					class_precision = np.divide(tps, np.sum([tps, fps]))
					class_recall = np.divide(tps, np.sum([tps, fns]))
					class_f1_score = np.divide(np.multiply(2, np.multiply(class_precision, class_recall)),
											   np.sum([class_precision, class_recall]))
				else:
					class_precision = 0
					class_recall = 0
					class_f1_score = 0
				per_class_scores.append((class_id, num_examples, class_precision, class_recall, class_f1_score))


			metis_logger.info(msg="{}, Total Distributed Validation Samples: {}".format(learner_id, str(
				np.sum([x[1] for x in per_class_scores]))))
			metis_logger.info(
				msg="{}, Scores Per Class (Precision, Recall, F1): {}".format(learner_id, per_class_scores))

			macro_weighted_federation_f1_score = np.divide(
				np.sum([x[1] * x[4] for x in per_class_scores]),
				np.sum([x[1] for x in per_class_scores]))
			metis_logger.info(
				msg="{}, Macro-Weighted F1-Score: {}".format(learner_id, str(macro_weighted_federation_f1_score)))

			micro_federation_f1_score = np.divide(np.multiply(2, cumulative_tps), np.sum(
				[np.multiply(2, cumulative_tps), cumulative_fps, cumulative_fns]))
			metis_logger.info(msg="{}, Micro F1-Score: {}".format(learner_id, str(micro_federation_f1_score)))

			# If a classifier makes zero predictions for a particular class then the Geometric Mean is 0.
			# To alleviate this problem, we enforce a small positive correction value: 0.001.
			gmean_recalls = np.array([x[3] for x in per_class_scores])
			metis_logger.info(msg="{}, G-Mean Recalls (original): {}".format(learner_id, gmean_recalls))
			gmean_recalls[gmean_recalls == 0.0] = 0.001
			metis_logger.info(msg="{}, G-Mean Recalls (altered): {}".format(learner_id, gmean_recalls))
			gmean_score = np.power(np.product(gmean_recalls), np.divide(1, len(gmean_recalls)))
			metis_logger.info(msg="{}, G-Mean Score: {}".format(learner_id, str(gmean_score)))


		if self.metis_db_session.is_regression:
			total_number_of_elements = np.sum([fedeval.validation_eval_metrics['num_examples']
											   for fedeval in federation_evaluations])

			federation_se_sum = np.sum(
				[np.multiply(fedeval.validation_eval_metrics['mse'], fedeval.validation_eval_metrics['num_examples'])
				 for fedeval in federation_evaluations])
			federation_mse = np.divide(federation_se_sum, total_number_of_elements)
			imse = np.divide(1, federation_mse)

			federation_ae_sum = np.sum(
				[np.multiply(fedeval.validation_eval_metrics['mae'], fedeval.validation_eval_metrics['num_examples'])
				 for fedeval in federation_evaluations])
			federation_mae = np.divide(federation_ae_sum, total_number_of_elements)
			imae = np.divide(1, federation_mae)

		# The default weighting value is always the inverted federation validation loss.
		weighting_value = ifvl
		if self.community_function == "DVWMacroWeightedF1":
			weighting_value = macro_weighted_federation_f1_score
		elif self.community_function == "DVWMicroF1":
			weighting_value = micro_federation_f1_score
		elif self.community_function == "DVWInvertedMSE":
			weighting_value = imse
		elif self.community_function == "DVWInvertedMAE":
			weighting_value = imae

		""" G-MEAN """
		weighting_value = gmean_score

		# """ DELTA WEIGHTING """
		# # We keep track of the FederationDriver so make sure the learner_id belongs to actual learners!
		# if learner_id in self.learners_ids:
		# 	if learner_id not in self.delta_weighting_state_map:
		# 		# Collect all learners first distributed evaluation.
		# 		self.delta_weighting_state_map[learner_id] = weighting_value
		# 	else:
		# 		if len(self.delta_weighting_state_map.keys()) == len(self.learners_ids):
		# 			# Classification example (Learner, accuracy):
		# 			# 	Federation Round 1: (L1, 80%) (L2, 70%) (L3, 90%)
		# 			# 		Ebase = min(0.8, 0.7, 0.9) = 0.7
		# 			# 	Federation Round 2: (L1, 95%) (L2, 65%) (L3, 88%)
		# 			# 		Delta_L1 = max(0, (0.95 - 0.7)) = 0.25
		# 			# 		Delta_L2 = max(0, (0.65 - 0.7)) = 0
		# 			# 		Delta_L3 = max(0, (0.88 - 0.7)) = 0.18
		# 			# Regression example:
		# 			e_base = np.min(list(self.delta_weighting_state_map.values()))
		# 			delta_weighting = np.max([0, (weighting_value - e_base)])
		# 			weighting_value = delta_weighting
		# 			print("Delta Weighting: ", learner_id, delta_weighting)


		if is_community_model is False:
			# We normalize the learner's weight using the first ever recorded weighting value.
			# if len(self.learners_validation_requests_loss_state_map.values()) > 0 and \
			# 		learner_id in self.learners_validation_requests_loss_state_map.keys():
			# 	flatten_list = []
			# 	[flatten_list.extend(lst) for lst in self.learners_validation_requests_loss_state_map.values()]
			# 	sorted_weighting_values_by_time = sorted(flatten_list, key=lambda v: v[0])
			# 	sorted_weighting_values_by_time = [w[1] for w in sorted_weighting_values_by_time]
			# 	norm_weighting_value = sorted_weighting_values_by_time[0]
			# 	weighting_metric /= norm_weighting_value
			# metis_logger.info(msg="Learner {}, Normalized Federation Value: {}".format(learner_id, weighting_metric))
			final_federation_weighting_value = weighting_value

			self.__learners_validation_requests_loss_state_map[learner_id].append(
				(TimeUtil.current_milli_time(), final_federation_weighting_value))

			# Required output for Local Learner Model Learning Progress
			metis_logger.info(msg="Learner: {}, Model Evaluation on each Validation Set: {}".format(
				learner_id, federation_evaluations))
			metis_logger.info(
				msg="Learner: {}, Model Weighted Value: {}".format(learner_id, final_federation_weighting_value))

		else:
			# if len(self.community_model_validation_requests_loss) > 0:
			# 	community_weighting_values = [x[1] for x in self.community_model_validation_requests_loss]
			# 	weighting_metric /= community_weighting_values[0]
			# metis_logger.info(msg="Community Model Normalized Federation Loss Value: {}".format(weighting_metric))
			final_federation_weighting_value = weighting_value
			self.__community_model_validation_requests_loss.append(
				(TimeUtil.current_milli_time(), final_federation_weighting_value))

			# Required output for Community Model Learning Progress
			metis_logger.info(
				msg="Community Model Evaluation on each Validation Set: {}".format(federation_evaluations))
			metis_logger.info(msg="Community Model Weighted Value: {}".format(final_federation_weighting_value))

		return final_federation_weighting_value


	def __handle_learner_evaluation_request(self, learner_id, model_variables, is_community_model):

		with self.__model_eval_lock:
			with tf.device(self.__evaluation_graph_server_device_fn):
				with self.__dvw_evaluation_graph_ops.eval_graph.as_default():

					if is_community_model:
						sess = self.community_models_evaluation_graph_session
					else:
						sess = self.local_models_evaluation_graph_session

					for (index, model_variable) in enumerate(model_variables):
						sess.run(self.__dvw_evaluation_graph_ops.assign_atomic_feds_ops[index],
								 feed_dict={self.__dvw_evaluation_graph_ops.assign_atomic_fed_placeholders[index]:
												model_variable})

					start_time = TimeUtil.current_milli_time()
					federation_validation_weighting = []
					for host_id, validation_set_ops in self.__dvw_evaluation_graph_ops \
							.hosts_validation_dataset_ops.items():
						validation_dataset_init_op = validation_set_ops.dataset_init_op
						next_validation_dataset_op = validation_set_ops.dataset_next
						validation_dataset_size = validation_set_ops.dataset_size
						validation_batches_num = int(validation_dataset_size / self.batch_size)
						self.__dvw_evaluation_graph_ops.tf_eval_ops.assign_tfdatasets_operators(
							validation_init_op=validation_dataset_init_op,
							next_validation_dataset=next_validation_dataset_op)

						_, valid_eval_metrics, _ = self.__dvw_evaluation_graph_ops \
							.tf_eval_ops.evaluate_model_on_existing_graph(tf_session=sess,
																		  target_stat_name=self.target_stat_name,
																		  learner_id=learner_id,
																		  compute_losses=True,
																		  include_json_evaluation_per_class=True)
						validation_score = valid_eval_metrics[self.target_stat_name]
						federation_validation_weighting.append(SingleValidationResult(
							learner_id=host_id, validation_dataset_size=validation_dataset_size,
							validation_batches_num=validation_batches_num, validation_eval_metrics=valid_eval_metrics,
							validation_score_name=self.target_stat_name,
							validation_score=validation_score,
							validation_loss_sum=valid_eval_metrics['validation_loss_sum'],
							validation_loss_squared_sum=valid_eval_metrics['validation_loss_squared_sum'],
							validation_loss_mean=valid_eval_metrics['validation_loss_mean'],
							validation_loss_variance=valid_eval_metrics['validation_loss_variance']))

					final_federation_weighting_value = self.__compute_learner_weighting_value(
						learner_id, federation_validation_weighting, is_community_model)
					end_time = TimeUtil.current_milli_time()
					metis_logger.info(msg="Learner: {}, Model Evaluation Request Time(ms): {}"
									  .format(learner_id, end_time - start_time))

					self.evaluation_requests_meta.append(EvaluationRequestMeta(
						learner_id=learner_id, request_init_unix_time_ms=start_time, request_end_unix_time_ms=end_time,
						is_community_model=is_community_model,
						learner_validation_weight=final_federation_weighting_value,
						learner_federated_validation=federation_validation_weighting))

					return federation_validation_weighting, final_federation_weighting_value


	def ModelEvaluationOnFederationValidationSets(self, request_iterator, context):

		for request in request_iterator:

			# Get learner id
			learner_id = request.learner_execution_result.execution_metadata.learner.learner_id

			# Whether this is a community model or not
			is_community_model = request.is_community_model

			# Grasp learner model and construct ndarrays from protobuff network_matrices
			learner_network_matrices_pb = request.learner_execution_result.network_matrices.matrices
			model_variables_ndarrays = ProtoBuffExchangeOps.reconstruct_ndarrays_from_network_matrices_pb(
				network_matrices_pb=learner_network_matrices_pb)

			future = self.__thread_executor.submit(self.__handle_learner_evaluation_request, learner_id,
												   model_variables_ndarrays, is_community_model)

			# If the model under evaluation is a local model, then the evaluation needs to be done synchronously, i.e.,
			# FIFO - serializable evaluations. If the model under evaluation is a community model, then the evaluation
			# can be done asynchronously.
			learner_validation_weight = 1
			if is_community_model is False:
				federation_validation_weighting, learner_validation_weight = future.result()
			else:
				self.__community_model_validation_requests_futures.append(future)

			double_value_pb = ProtoBuffExchangeOps.construct_double_value_pb(learner_validation_weight)

			return double_value_pb


	def RetrieveValidationMetadataFromEvaluator(self, request, context):
		learner_id = request.learner_id

		# Loop through and execute any pending community model evaluation requests
		for future in self.__community_model_validation_requests_futures:
			if not future.done():
				try:
					result = future.result()
				except Exception as exc:
					print('An exception was generated for the community model evaluation request future: %s' % exc)

		json_result = self.toJSON_representation()
		jsonstring_pb = ProtoBuffExchangeOps.construct_json_string_value_pb(json_string=json_result)
		return jsonstring_pb


	def toJSON_representation(self):
		json_result = {"evaluation_requests": [obj.toJSON_representation() for obj in self.evaluation_requests_meta]}
		# Erase existing metadata for new ones.
		self.evaluation_requests_meta = list()
		json_result = json.dumps(json_result)
		return json_result


	def shutdown(self):
		self.local_models_evaluation_graph_session.close()
		self.community_models_evaluation_graph_session.close()


class GRPCEvaluator(GRPCServer):

	def __init__(self, model_evaluator_grpc_servicer, executor):
		GRPCServer.__init__(self, model_evaluator_grpc_servicer, thread_pool_executor=executor)

	def start(self):
		model_evaluation_serving_pb2_grpc.add_EvalServingServicer_to_server(
			servicer=self.grpc_servicer,
			server=self.server)
		self.server.add_insecure_port(self.grpc_servicer.grpc_servicer_host_port)
		try:
			self.server.start()
		except Exception as e:
			metis_logger.info(msg="Metis GRPC Model Evaluator service threw an exception: {}".format(e))
		try:
			while True:
				time.sleep(self.service_lifetime)
				metis_logger.info("Metis GRPC Model Evaluator service is still running")
		except KeyboardInterrupt:
			# TODO change this if more seconds are needed
			self.server.stop(grace=0)  # grace timeout of 200 secs, default was 0 (None)

	def stop(self):
		stopping_event = self.server.stop(grace=0)  # grace timeout of 0 (None)
		return stopping_event


class FedModelEvaluator(object):
	"""
	A helper class for initializing Model Evaluator as a Process
	"""

	def __init__(self, fed_environment, federated_variables, federation_model_obj,
				 metis_db_session, target_stat_name, max_workers):

		# We use a ThreadPoolExecutor to run our GRPC servicer since Tensorflow is not
		# thread safe when it is spawned from a child process and very strange behavior is seen
		# e.g.  1. thread interleaving between grpc handler and tf.session threads
		#       2. tf.session hangs on large sized variables.
		# See here: https://github.com/tensorflow/tensorflow/issues/14442
		self.__executor = ThreadPoolExecutor(max_workers=max_workers,
											 thread_name_prefix='PoolExecutorOf_{}'.format("GRPCEvaluator"))
		self.__model_evaluator_grpc_servicer = ModelEvaluatorServicer(fed_environment=fed_environment,
																	  federated_variables=federated_variables,
																	  federation_model_obj=federation_model_obj,
																	  metis_db_session=metis_db_session,
																	  target_stat_name=target_stat_name,
																	  executor=self.__executor)
		self.__model_evaluator_grpc_server = GRPCEvaluator(self.__model_evaluator_grpc_servicer, self.__executor)

		self.grpc_server_future = None

	def start(self):
		metis_logger.info(msg='Initializing GRPC Metis Evaluator.')
		# If evaluator instance is set as daemon then the thread pool will not close on program exit
		self.grpc_server_future = self.__executor.submit(self.__model_evaluator_grpc_server.start)
		metis_logger.info(msg='GRPC Metis Evaluator Initialized @ {}'.format(
			self.__model_evaluator_grpc_servicer.grpc_servicer_host_port))
		time.sleep(0.1)  # Wait till the Evaluation Server is fully initialized

	def stop(self):
		metis_logger.info(msg='Shutting down GRPC Metis Evaluator.')
		self.__model_evaluator_grpc_servicer.shutdown()
		self.__model_evaluator_grpc_server.stop()

		# TODO Following this is a hack in order to signal that the thread pool executor needs to shut down.
		#  Check solution at stackoverflow:
		#  https://stackoverflow.com/questions/48350257/how-to-exit-a-script-after-threadpoolexecutor-has-timed-out
		try:
			self.grpc_server_future.result(timeout=1)
		except TimeoutError as time_error:
			import atexit
			atexit.unregister(concurrent.futures.thread._python_exit)
			self.__executor.stop = lambda wait: None

		metis_logger.info(msg='GRPC Metis Evaluator shut down.')
