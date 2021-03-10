import math
import os
import random
import threading
import time

import federation.fed_cluster_env as fedenv
import federation.fed_model as fedmodel
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import defaultdict
from federation.fed_grpc_controller_client import FedClient
from federation.fed_grpc_controller import FedController
from federation.fed_grpc_evaluator import FedModelEvaluator
from federation.fed_grpc_evaluator_client import FedModelEvaluatorClient
from metisdb.metisdb_session import MetisDBSession
from scipy import stats
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.tf.tf_ops_evaluation import TFGraphEvaluation
from utils.tf.tf_ops_configs import TFConfiguration
from utils.generic.time_ops import TimeUtil


scriptDirectory = os.path.dirname(os.path.realpath(__file__))


class FedRoundExecutionResults(object):

	def __init__(self, metis_grpc_controller_metadata, metis_grpc_evaluator_metadata,
				 hosts_metadata_results, fedround_execution_time_ms, fedround_evaluation):
		self.metis_grpc_controller_metadata = metis_grpc_controller_metadata
		self.metis_grpc_evaluator_metadata = metis_grpc_evaluator_metadata
		self.hosts_metadata_results = hosts_metadata_results
		self.fedround_execution_time_ms = fedround_execution_time_ms
		self.fedround_evaluation = fedround_evaluation

	def __str__(self):
		return "Hosts Execution Metadata Results: {}, Federation Round Execution Time in ms: {}, " \
			   "Federation Round Evaluation: {}".format(self.hosts_metadata_results, self.fedround_execution_time_ms,
														self.fedround_evaluation)

	def toJSON_representation(self):
		return {'hosts_results': self.hosts_metadata_results,
				'fedround_execution_time_ms': self.fedround_execution_time_ms,
				'fedround_evaluation': self.fedround_evaluation,
				'metis_grpc_controller_metadata': self.metis_grpc_controller_metadata,
				'metis_grpc_evaluator_metadata': self.metis_grpc_evaluator_metadata}


class FedHostExecutionResultsStats(object):

	def __init__(self, host_id, training_devices, completed_epochs, completed_batches, num_training_examples,
				 num_validation_examples, compute_init_unix_time=None, compute_end_unix_time=None,
				 test_set_evaluations=list(), train_set_evaluations=list(), validation_set_evaluations=list(),
				 epochs_exec_times_ms=list(), epochs_proc_times_ms=list(), metis_grpc_client_metadata=None,
				 extra_measurements=None):
		self.host_id = host_id
		self.training_devices = training_devices
		self.completed_epochs = completed_epochs
		self.completed_batches = completed_batches
		self.num_training_examples = num_training_examples
		self.num_validation_examples = num_validation_examples
		self.compute_init_unix_time = compute_init_unix_time
		self.compute_end_unix_time = compute_end_unix_time
		self.test_set_evaluations = test_set_evaluations
		self.train_set_evaluations = train_set_evaluations
		self.validation_set_evaluations = validation_set_evaluations
		self.epochs_exec_times_ms = epochs_exec_times_ms
		self.epochs_proc_times_ms = epochs_proc_times_ms
		self.metis_grpc_client_metadata = metis_grpc_client_metadata
		self.extra_measurements = extra_measurements

	def __str__(self):
		return "Host ID: {}, Device ID: {}, Completed Epochs: {}, Completed Batches: {}, Local Dataset Size: {}, " \
			   "Epochs Execution Times: {}, Test Set Evaluations: {}".format(self.host_id, self.training_devices,
																			 self.completed_epochs,
																			 self.completed_batches,
																			 self.num_training_examples,
																			 self.epochs_exec_times_ms,
																			 self.test_set_evaluations)

	def toJSON_representation(self):
		return {'host_id': self.host_id, 'training_devices': self.training_devices,
				'completed_epochs': self.completed_epochs, 'completed_batches': self.completed_batches,
				'num_training_examples': self.num_training_examples,
				'num_validation_examples': self.num_validation_examples,
				'compute_init_unix_time': self.compute_init_unix_time,
				'compute_end_unix_time': self.compute_end_unix_time, 'test_set_evaluations': self.test_set_evaluations,
				'train_set_evaluations': self.train_set_evaluations,
				'validation_set_evaluations': self.validation_set_evaluations,
				'epochs_exec_times_ms': self.epochs_exec_times_ms,
				'epochs_proc_times_ms': self.epochs_proc_times_ms,
				'grpc_client_metadata': self.metis_grpc_client_metadata,
				'extra_measurements': self.extra_measurements}


class FedHostTempEval(object):

	def __init__(self, unix_ms=None, eval_results=None, is_after_epoch_completion=False,
				 is_after_community_update=False):
		self.unix_ms = unix_ms
		self.evaluation_results = eval_results
		self.is_after_epoch_completion = is_after_epoch_completion
		self.is_after_community_update = is_after_community_update


class FedExecutionOps(object):


	@classmethod
	def federated_between_graph_replication(cls,
											fed_environment,
											federation_model_obj,
											metis_db_session,
											session_target_stat_name,
											batch_level_log=False) -> OrderedDict:
		"""
		This function performs a tensorflow between graph replication for a total number of # rounds. The federation
		model is being shipped to each worker and each worker trains its variables (biases/weights) locally. Once each
		round finishes we calculate the average of each variable.
		:param fed_environment:
		:param federation_model_obj:
		:param metis_db_session:
		:param session_target_stat_name:
		:param batch_level_log
		:return
		"""

		if not isinstance(fed_environment, fedenv.FedEnvironment):
			raise TypeError("`fed_environment` must be of type %s " % fedenv.FedEnvironment)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s " % fedmodel.FedModelDef)

		federated_variables = fedmodel.FedModelDef.construct_model_federated_variables(federation_model_obj)

		# Initialize federation with specific weights.
		# npzfile = np.load(scriptDirectory + "/clients_5F_Function_AsyncFedAvg_Second50PctData_60mins_model_run1.npz")
		# npzfile = np.load("/tmp/metis_project/execution_logs/federation_models/community_model_federation_round_24.npz")
		# arrays_ids = list(npzfile.keys())
		# for idx in range(len(federated_variables)):
		# 	array_value = npzfile[arrays_ids[idx]]
		# 	federated_variables[idx].value = array_value

		federation_rounds = fed_environment.federation_rounds
		fed_hosts = fed_environment.fed_training_hosts

		synchronous_execution = fed_environment.synchronous_execution
		controller_grpc_servicer_host_port = fed_environment.federation_controller_grpc_servicer_endpoint
		evaluator_grpc_servicer_host_port = fed_environment.federation_evaluator_grpc_servicer_endpoint
		community_function = fed_environment.community_function
		target_exec_time_mins = fed_environment.execution_time_in_mins

		metis_logger.info(msg='Synchronous Execution: %s' % synchronous_execution)
		federated_execution_st = TimeUtil.current_milli_time()
		fedrounds_execution_results = OrderedDict()

		# Initialize Federation Controller and register all learners.
		required_learners_for_community_update = 1
		fed_controller_workers = 20
		participating_hosts_ids = [host.host_identifier for host in fed_environment.fed_training_hosts]
		fed_controller = FedController(grpc_servicer_host_port=controller_grpc_servicer_host_port,
									   participating_hosts_ids=participating_hosts_ids,
									   synchronous_execution=synchronous_execution,
									   target_learners=None,
									   target_epochs=None,
									   target_score=None,
									   target_exec_time_mins=target_exec_time_mins,
									   required_learners_for_community_update=required_learners_for_community_update,
									   max_workers=fed_controller_workers)
		fed_controller.start()

		# Initialize Federation Evaluator.
		fed_evaluator_workers = 20
		fed_evaluator = FedModelEvaluator(fed_environment=fed_environment,
										  federated_variables=federated_variables,
										  federation_model_obj=federation_model_obj,
										  metis_db_session=metis_db_session,
										  target_stat_name=session_target_stat_name,
										  max_workers=fed_evaluator_workers)
		fed_evaluator.start()


		# This grpc driver does not listen to any signal, it just submits requests to the federation controller.
		driver_controller_grpc_client = FedClient(client_id="FederationControllerDriver",
												  controller_host_port=controller_grpc_servicer_host_port)
		driver_evaluator_grpc_client = FedModelEvaluatorClient(client_id="FederationEvaluatorDriver",
															   evaluator_host_port=evaluator_grpc_servicer_host_port)
		hosts_execution_metadata_map = dict()

		# Start federated training.
		for frid in range(federation_rounds):

			federation_round_st = TimeUtil.current_milli_time()
			metis_logger.info(msg='Federation Running Round: %d' % (frid + 1))
			previous_hosts_execution_metadata_map = hosts_execution_metadata_map
			hosts_execution_metadata_map = dict()

			# We create and run one thread for each remote host.
			fedtraining_threads = list()

			for lidx, fed_host in enumerate(fed_hosts):
				fed_host_local_cluster_spec = fed_host.local_cluster_spec
				metis_logger.info(fed_host_local_cluster_spec)
				thread_name = '{}/FedTrainRound/{}'.format(fed_host.host_identifier, frid)

				########################################### SEMI-SYNC ########################################
				SEMI_SYNCHRONOUS_EXECUTION = str(os.environ["SEMI_SYNCHRONOUS_EXECUTION"]) == "True" \
					if "SEMI_SYNCHRONOUS_EXECUTION" in os.environ else False

				if SEMI_SYNCHRONOUS_EXECUTION:					
					# For the first round we allow the learners to perform a cold start.
					# The learner thread will set this value upon invocation.
					if len(previous_hosts_execution_metadata_map.keys()) == 0:
						# This is for cold start. The learner thread will set this value.
						FEDHOST_NUMBER_OF_BATCHES = None
					else:
						tmax = -1
						hosts_time_per_batch = dict()
						for host_id, exec_result in previous_hosts_execution_metadata_map.items():
							# In case the local training dataset is not entirely divisible by the batch size.
							host_id_fed_host_obj = \
								[fed_host_obj for fed_host_obj in fed_hosts if fed_host_obj.host_identifier==host_id][0]
							host_id_number_of_batches_per_epoch = int(np.ceil(np.divide(
								exec_result.num_training_examples,
								host_id_fed_host_obj.fed_worker_servers[0].batch_size)))
							if 'GPU' in host_id_fed_host_obj.host_training_devices:
								time_per_batch_ms = float(os.environ["GPU_TIME_PER_BATCH_MS"])
							else:
								time_per_batch_ms = float(os.environ["CPU_TIME_PER_BATCH_MS"])
							hosts_time_per_batch[host_id] = time_per_batch_ms
							epoch_proc_time_ms = np.multiply(host_id_number_of_batches_per_epoch, time_per_batch_ms)
							tmax = np.max([tmax, epoch_proc_time_ms])

						SEMI_SYNCHRONOUS_K_VALUE = float(os.environ["SEMI_SYNCHRONOUS_K_VALUE"])
						tmax = np.multiply(SEMI_SYNCHRONOUS_K_VALUE, tmax)
						FEDHOST_NUMBER_OF_BATCHES = \
							int(np.floor(np.divide(tmax, hosts_time_per_batch[fed_host.host_identifier])))

					t = threading.Thread(target=cls.__semisync_fedhost_training,
										 name=thread_name,
										 args=[FEDHOST_NUMBER_OF_BATCHES,
											   fed_host,
											   federated_variables,
											   federation_model_obj,
											   community_function,
											   metis_db_session,
											   controller_grpc_servicer_host_port,
											   evaluator_grpc_servicer_host_port,
											   hosts_execution_metadata_map,
											   session_target_stat_name,
											   batch_level_log],
										 daemon=True)
					fedtraining_threads.append(t)
				###################################### END OF SEMI-SYNC ########################################

				elif synchronous_execution:
					t = threading.Thread(target=cls.__sync_fedhost_training,
										 name=thread_name,
										 args=[frid, lidx,
											   fed_host,
											   federated_variables,
											   federation_model_obj,
											   community_function,
											   metis_db_session,
											   controller_grpc_servicer_host_port,
											   evaluator_grpc_servicer_host_port,
											   hosts_execution_metadata_map,
											   session_target_stat_name,
											   batch_level_log],
										 daemon=True)
					fedtraining_threads.append(t)
				else:
					t = threading.Thread(target=cls.__async_fedhost_training,
										 name=thread_name,
										 args=[fed_host,
											   federated_variables,
											   federation_model_obj,
											   community_function,
											   metis_db_session,
											   controller_grpc_servicer_host_port,
											   evaluator_grpc_servicer_host_port,
											   hosts_execution_metadata_map,
											   session_target_stat_name,
											   batch_level_log],
										 daemon=True)
					fedtraining_threads.append(t)

			for t in fedtraining_threads:
				t.start()

			for t in fedtraining_threads:
				t.join()

			# Request community update from the federation controller and update the federated variables collection.
			community_update_matrices = driver_controller_grpc_client\
				.request_current_community(send_learner_state=False, block=True)

			for idx in range(len(federated_variables)):
				federated_variables[idx].value = community_update_matrices[idx]

			# Store the latest community model
			# output_directory = "/tmp/metis_project/execution_logs/federation_models"
			# model_output_file = output_directory + '/community_model_federation_round_{}.npz'.format(frid)
			# np.savez(model_output_file, *community_update_matrices)

			# If weighting scheme is DVW, then evaluate community model on the federation validation dataset.
			# This request is issued in order to log the performance of the community model.
			if 'DVW' in community_function:
				driver_evaluator_grpc_client.request_model_evaluation(model_variables=community_update_matrices,
																	  is_community_model=True,
																	  block=True)

			# TODO Model evaluation here is assigned to the first host, it could be any host!
			# Test model efficiency using the latest community weights.
			eval_results = cls.evaluate_federation_model_on_remote_host(
				fed_host_obj=fed_hosts[0], federation_model_obj=federation_model_obj, metis_db_session=metis_db_session,
				federated_variables=federated_variables, session_target_stat_name=session_target_stat_name)
			metis_logger.info("Federation Model Evaluation Metrics: {}".format(eval_results))

			metis_logger.info("Federation Round Completed Local Batches: {}".format(
				np.sum([hosts_execution_metadata_map[host_id].completed_batches for host_id in hosts_execution_metadata_map])
			))

			current_stat_score = eval_results[session_target_stat_name]
			temp_eval = FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
										eval_results=eval_results,
										is_after_community_update=True)

			federation_round_et = TimeUtil.current_milli_time()
			federation_round_dt = TimeUtil.delta_diff_in_ms(federation_round_et, federation_round_st)

			# End time of entire federated execution.
			federated_execution_et = TimeUtil.current_milli_time()
			federated_execution_dt = TimeUtil.delta_diff_in_mins(federated_execution_st, federated_execution_et)

			# The following is used just for storing metadata necessary for future analysis
			dict_key = 'federation_round_{}'.format(frid)

			metis_controller_metadata = driver_controller_grpc_client.retrieve_latest_fedround_controller_metadata()
			metis_evaluator_metadata = driver_evaluator_grpc_client.retrieve_evaluator_metadata()

			fedrounds_execution_results[dict_key] = FedRoundExecutionResults(
				metis_grpc_controller_metadata=metis_controller_metadata,
				metis_grpc_evaluator_metadata=metis_evaluator_metadata,
				hosts_metadata_results=hosts_execution_metadata_map,
				fedround_execution_time_ms=federation_round_dt,
				fedround_evaluation=temp_eval)

			# Check against score signal.
			if driver_controller_grpc_client.is_target_stat_score_reached(current_stat_score):
				metis_logger.info(msg="Above target score. Breaking at round: %s " % (frid + 1))
				break

			# Check against execution time signal.
			if driver_controller_grpc_client.is_target_exec_time_reached(federated_execution_dt):
				metis_logger.info(msg="Above target execution time. Breaking at round: %s " % (frid + 1))
				break

			driver_controller_grpc_client.notify_controller_to_reset()

		metis_logger.info(msg="Total Federation Execution Time: %.3f mins" % federated_execution_dt)

		# Shutdown Federation Evaluator and Controller.
		fed_evaluator.stop()
		fed_controller.stop()

		return fedrounds_execution_results


	@classmethod
	def __sync_fedhost_training(cls,
								frid, lidx,
								fed_host_obj,
								federated_variables,
								federation_model_obj,
								community_function,
								metis_db_session,
								controller_host_port,
								evaluator_host_port,
								hosts_execution_metadata_map,
								session_target_stat_name,
								batch_level_log=False):
		"""
		:param fed_host_obj
		:param federated_variables:
		:param federation_model_obj:
		:param metis_db_session:
		:param controller_host_port:
		:param hosts_execution_metadata_map:
		:param session_target_stat_name
		:param logging:
		:return:
		"""

		if not isinstance(fed_host_obj, fedenv.FedHost):
			raise TypeError("The provided fed_host_obj parameter must be of type %s" % fedenv.FedHost)
		if not any(isinstance(fedvar, fedmodel.FedVar) for fedvar in federated_variables):
			raise TypeError("All the federated variables passed to this function must be of type %s " % fedmodel.FedVar)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s" % fedmodel.FedModelDef)

		tf_cluster_spec = fed_host_obj.cluster_spec
		tf_cluster_master = fed_host_obj.fed_master
		tf_worker_server = fed_host_obj.fed_worker_servers[0]
		w_device_name = tf_worker_server.device_name
		w_is_remote_server = tf_worker_server.is_remote_server
		w_is_chief = tf_worker_server.is_leader
		w_gpu_id = tf_worker_server.gpu_id
		w_local_batch_size = tf_worker_server.batch_size
		w_local_epochs = tf_worker_server.target_update_epochs
		host_id = fed_host_obj.host_identifier
		host_training_devices = fed_host_obj.host_training_devices
		w_is_fast_learner = True if "GPU" in host_training_devices else False
		w_config = TFConfiguration.tf_session_config(is_gpu=w_is_fast_learner)

		if w_is_remote_server:
			w_device = tf.train.replica_device_setter(cluster=tf_cluster_spec, worker_device=w_device_name)
		else:
			w_device = '/gpu:0' if w_gpu_id is None else '/gpu:{}'.format(w_gpu_id)

		compute_init_time = TimeUtil.current_milli_time()
		temporal_test_evaluations = list()
		temporal_train_evaluations = list()
		temporal_validation_evaluations = list()

		# Separate the 'DVW' community weighting scheme/function from other functions (e.g., 'FedAvg').
		dvw_weighting_scheme = True if "DVW" in community_function else False
		host_completed_epochs = 0
		host_completed_batches = 0
		epochs_times = list()

		host_controller_grpc_client = FedClient(client_id=host_id, controller_host_port=controller_host_port)

		# TODO The evaluation request needs to be moved to the controller end.
		host_evaluator_grpc_client = FedModelEvaluatorClient(client_id=host_id, evaluator_host_port=evaluator_host_port)

		with tf.device(w_device):
			tf.reset_default_graph()
			exec_graph = tf.Graph()
			with exec_graph.as_default():

				# To ensure randomly deterministic operations and values, we set the random seed.
				np.random.seed(seed=1990)
				random.seed(1990)
				tf.set_random_seed(seed=1990)

				# Import all learner datasets structure.
				train_dataset_structure, validation_dataset_structure, test_dataset_structure = metis_db_session\
					.import_host_data(learner_id=host_id,
									  batch_size=w_local_batch_size,
									  import_train=True,
									  import_validation=True,
									  import_test=True)

				training_init_op = train_dataset_structure.dataset_init_op
				next_train_dataset = train_dataset_structure.dataset_next
				training_dataset_size = train_dataset_structure.dataset_size

				validation_init_op = validation_dataset_structure.dataset_init_op
				next_validation_dataset = validation_dataset_structure.dataset_next
				validation_dataset_size = validation_dataset_structure.dataset_size

				testing_init_op = test_dataset_structure.dataset_init_op
				next_test_dataset = test_dataset_structure.dataset_next
				testing_dataset_size = test_dataset_structure.dataset_size

				# Inform GRPC client regarding the number of training examples for this learner.
				host_controller_grpc_client.update_num_training_examples(val=training_dataset_size)
				host_controller_grpc_client.update_num_validation_examples(val=validation_dataset_size)
				host_controller_grpc_client.update_batch_size(val=w_local_batch_size)

				# Retrieve model's input & output placeholders.
				_x_placeholders = federation_model_obj.input_tensors_datatype()
				_y_placeholders = federation_model_obj.output_tensors_datatype()

				# Define model's global step.
				_global_step = tf.train.get_or_create_global_step()

				# Define Deep NN Graph.
				model_architecture = federation_model_obj.model_architecture(input_tensors=_x_placeholders,
																			 output_tensors=_y_placeholders,
																			 global_step=_global_step,
																			 batch_size=w_local_batch_size,
																			 dataset_size=training_dataset_size)

				# Retrieve FedOperation and FedTensor collections, see projectmetis/federation/fed_model.py.
				train_step = model_architecture.train_step
				loss_tensor_fedmodel = model_architecture.loss
				loss_tensor = loss_tensor_fedmodel.get_tensor()
				predictions_tensor_fedmodel = model_architecture.predictions

				# Register tensorflow evaluation operations based on evaluation task: regression or classification.
				tf_graph_evals = TFGraphEvaluation(_x_placeholders, _y_placeholders, predictions_tensor_fedmodel,
												   is_classification=metis_db_session.is_classification,
												   is_regression=metis_db_session.is_regression,
												   num_classes=metis_db_session.num_classes,
												   negative_classes_indices=metis_db_session.negative_classes_indices,
												   is_eval_output_scalar=metis_db_session.is_eval_output_scalar)
				tf_graph_evals.assign_tfdatasets_operators(training_init_op, next_train_dataset, validation_init_op,
														   next_validation_dataset, testing_init_op, next_test_dataset)
				tf_graph_evals.register_evaluation_ops(metis_db_session.get_learner_evaluation_output_attribute(host_id),
													   w_local_batch_size)

				# Get the trainable (federated) variables collection defined in the model graph.
				trainable_variables = model_architecture.model_federated_variables

				# There could be a case that the testing variables are different form the training variables,
				# 	e.g. TRAINABLE collection vs MOVING_AVERAGE collection.
				# Based on this case, we take one additional step to define which variables are
				# the 'true' model's evaluation variables.
				udf_testing_vars = model_architecture.user_defined_testing_variables_collection
				if udf_testing_vars is not None:
					udf_testing_vars = TFConfiguration.tf_trainable_variables_config(
						graph_trainable_variables=trainable_variables,
						user_defined_variables_collection=udf_testing_vars)

				"""			
				The `fedvars_atomic` variables scope will be stored in the Parameter Server(PS) and shared across 
				Workers ~ thus they must be declared as GLOBAL_VARIABLES. Their purpose is to initialize the weights & 
				biases of every Worker's model. The 'reuse=tf.AUTO_REUSE' initializes the variables if they do not 
				exist, otherwise returns their value.
				"""
				with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
					for idx, variable in enumerate(federated_variables):
						tf.get_variable(name=variable.name, initializer=variable.value,
										collections=variable.tf_collection, trainable=variable.trainable)

				# Define operations for model weights and biases initialization with the federated variables values.
				fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
				assign_atomic_feds = []
				for (index, model_var) in enumerate(trainable_variables):
					fed_var = fed_vars[index]
					assign_atomic_feds.append(tf.assign(model_var, fed_var))

				# A verification operation to make sure that all model variables have been initialized.
				uninitialized_vars = tf.report_uninitialized_variables()

				# model_saver_hook = tf.train.CheckpointSaverHook(
				# 	checkpoint_dir='/nfs/isd/stripeli/metis_execution_models_checkpoint_dir/',
				# 	save_secs=None,
				# 	save_steps=w_local_epochs*int(np.floor(np.divide(training_dataset_size, w_local_batch_size))),
				# 	saver=tf.train.Saver(),
				# 	checkpoint_basename='{}.model.ckpt'.format(data_partition_id),
				# 	scaffold=None)

				# TODO Reduce delay in terms of workers training initialization, see below:
				""" 
				Why there is delay between workers? Because Monitored Training Session checks whether the variables 
				in the underlying graph have been initialized or not. If the variables are not initialized yet an 
				warning/info will be printed as follows and a racing between the workers will occur:
					INFO:tensorflow:Waiting for model to be ready.  Ready_for_local_init_op:  Variables not initialized: 
				Then the MonitoredTrainingSession will put the 'uninitialized' worker(s) to sleep for 30 seconds and 
				repeat. Further reading: 
				https://stackoverflow.com/questions/43084960/
					tensorflow-variables-are-not-initialized-using-between-graph-replication
				"""

				# The 'is_chief' field in the MonitoringTrainingSession indicates whether we want a Master or a Worker
				# tensorflow graph training session.
				tf_master_grpc = tf_cluster_master.grpc_endpoint
				with tf.train.MonitoredTrainingSession(master=tf_master_grpc, is_chief=w_is_chief, config=w_config) \
						as mon_sess:

					# As long as we have uninitialized variables, we wait for the TF server to initialize all.
					while len(mon_sess.run(uninitialized_vars)) > 0:
						metis_logger.info(msg="Host TF Cluster %d: waiting for variables initialization..." % host_id)
						time.sleep(1.0)

					metis_logger.info("Host TF Cluster {}: Idle time due to Tensorflow graph operations creation "
									  "(secs): {}".format(host_id, TimeUtil.delta_diff_in_secs(
						TimeUtil.current_milli_time(), compute_init_time)))

					# Assign the community model to the learner.
					mon_sess.run(assign_atomic_feds)


					# TODO HACK!!
					# Evaluate Community Model on local dataset
					train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals \
						.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

					metis_logger.info("Host ID: {}, Community Model Train Evaluation Results: {}".format(
						host_id, train_eval_results))
					metis_logger.info("Host ID: {}, Community Model Validation Evaluation Results: {}".format(
						host_id, validation_eval_results))
					metis_logger.info("Host ID: {}, Community Model Test Evaluation Results: {}".format(
						host_id, test_eval_results))

					temporal_train_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	  eval_results=train_eval_results,
																	  is_after_community_update=True))
					temporal_validation_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																		   eval_results=validation_eval_results,
																		   is_after_community_update=True))
					temporal_test_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	 eval_results=test_eval_results,
																	 is_after_community_update=True))



					# Start local epoch training.
					for worker_epoch in range(w_local_epochs):

						mon_sess.run(training_init_op)
						epoch_current_batch_num = 0
						epoch_start_time = TimeUtil.current_milli_time()

						while True:
							try:

								# We always check if any federation round signal is reached before start training on a
								# new batch. If a signal is reached, then we need to record all host metadata info.
								if host_controller_grpc_client.is_federation_round_signal_reached:

										metis_logger.info(msg="Learner: {}, Federation round terminated, "
															  "wrapping up computation".format(host_id))

										# Retrieve latest trained weights/biases.
										final_trainable_vars = mon_sess.run(trainable_variables)

										# Update client completed epochs.
										host_controller_grpc_client.update_completed_epochs(val=host_completed_epochs)

										# Update client variables.
										host_controller_grpc_client.update_trained_variables(
											new_variables=final_trainable_vars)

										# Send local variables to controller. Blocking call because the thread will die.
										host_controller_grpc_client.send_model_local_trained_variables_to_controller(
											block=True)

										# Capture host metadata.
										host_metadata_result = FedHostExecutionResultsStats(
											host_id=host_id, training_devices=host_training_devices,
											completed_epochs=host_completed_epochs,
											completed_batches=host_completed_batches,
											num_training_examples=training_dataset_size,
											num_validation_examples=validation_dataset_size,
											compute_init_unix_time=compute_init_time,
											compute_end_unix_time=TimeUtil.current_milli_time(),
											test_set_evaluations=temporal_test_evaluations,
											train_set_evaluations=temporal_train_evaluations,
											validation_set_evaluations=temporal_validation_evaluations,
											epochs_exec_times_ms=epochs_times,
											metis_grpc_client_metadata=host_controller_grpc_client
												.toJSON_representation())

										# TODO extend with grpc client metadata
										# Update the execution HashMap of all learners.
										hosts_execution_metadata_map[host_id] = host_metadata_result
										host_controller_grpc_client.shutdown()

										# End of federation round training.
										return

								# Load the next training batch.
								train_batch = mon_sess.run(next_train_dataset)
								train_extra_feeds = OrderedDict()
								# We need to retrieve any additional placeholders declared during the construction of
								# the network. In this step we need to get the placeholders for the federated model
								# training operation. e.g. {'is_training': True}
								train_extra_feeds.update(train_step.get_feed_dictionary())
								for placeholder_name, placeholder_def in _x_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
								for placeholder_name, placeholder_def in _y_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

								# Train model (train_step FedOperation).
								train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)
								epoch_current_batch_num += 1
								host_completed_batches += 1

								# Update grpc client's number of completed batches.
								host_controller_grpc_client.update_completed_batches(val=host_completed_batches)

								# If logging is set, print every 10 batches.
								if batch_level_log and epoch_current_batch_num % 10 == 0:

									# Fetch the current global step.
									current_step = mon_sess.run(_global_step)
									if loss_tensor is not None:
										train_extra_feeds.update(predictions_tensor_fedmodel.get_feed_dictionary())
										train_loss = loss_tensor.eval_tf_tensor(session=mon_sess,
																				extra_feeds=train_extra_feeds)
										metis_logger.info(msg='Host: %s, Training Devices: %s, epoch %d, batch %d, '
															  'global step %d, loss %g'
															  % (host_id, host_training_devices, worker_epoch + 1,
																 epoch_current_batch_num, current_step, train_loss))

							except tf.errors.OutOfRangeError:
								break

						host_completed_epochs += 1

						# Update grpc client's local epoch counter.
						host_controller_grpc_client.update_completed_epochs(val=host_completed_epochs)

						epoch_end_time = TimeUtil.current_milli_time()

						# Compute training wall clock time (in milliseconds), store value and update grpc client.
						epochs_times.append(TimeUtil.delta_diff_in_ms(epoch_start_time, epoch_end_time))
						host_controller_grpc_client.update_processing_ms_per_epoch(val=np.mean(epochs_times))

						# Update controller that the learner just finished a training epoch.
						host_controller_grpc_client.notify_controller_for_fedround_signals(finished_training=False,
																						   finished_epoch=True,
																						   block=True)

						# Since the variables we need to use to test our model could be the ones defined by the user
						# (e.g. MovingAverage variables) we need compute the final variables values and then re-assign
						# them to the network trainable variables.
						final_trainable_vars = mon_sess.run(trainable_variables)

						# Update grpc client's variables with trained variables.
						host_controller_grpc_client.update_trained_variables(new_variables=final_trainable_vars)

						# If we have different testing variables from trainable variables, then
						# update the model with the testing variables and proceed with model evaluation
						if udf_testing_vars is not None:
							final_testing_vars = mon_sess.run(udf_testing_vars)
							for index, final_var_value in enumerate(final_testing_vars):
								fed_vars[index].load(final_var_value, mon_sess)
							mon_sess.run(assign_atomic_feds)

						# When to evaluate. In order to speed up training and experiments required execution time, it is
						# better to evaluate at the end of the local training cycle instead of every epoch.
						if worker_epoch == w_local_epochs-1:
							train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals\
								.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

							metis_logger.info("Host ID: {}, Local Model Train Evaluation Results: {}".format(
								host_id, train_eval_results))
							metis_logger.info("Host ID: {}, Local Model Validation Evaluation Results: {}".format(
								host_id, validation_eval_results))
							metis_logger.info("Host ID: {}, Local Model Test Evaluation Results: {}".format(
								host_id, test_eval_results))

							temporal_train_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																			  eval_results=train_eval_results,
																			  is_after_epoch_completion=True))
							temporal_validation_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																				   eval_results=validation_eval_results,
																				   is_after_epoch_completion=True))
							temporal_test_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																			 eval_results=test_eval_results,
																			 is_after_epoch_completion=True))

							latest_train_accuracy = train_eval_results[session_target_stat_name]
							host_controller_grpc_client.update_latest_train_score(val=latest_train_accuracy)
							latest_validation_score = validation_eval_results[session_target_stat_name]
							host_controller_grpc_client.update_latest_validation_score(val=latest_validation_score)

						# If the testing variables are defined then the network holds these testing variables, and now
						# we need to assign the trainable variables in order to proceed with the training.
						if udf_testing_vars is not None:
							for index, final_var_value in enumerate(final_trainable_vars):
								fed_vars[index].load(final_var_value, mon_sess)
								federated_variables[index].value = final_var_value
							mon_sess.run(assign_atomic_feds)

					if dvw_weighting_scheme:
						local_model_evaluation_score = host_evaluator_grpc_client.request_model_evaluation(
							model_variables=final_trainable_vars, is_community_model=False, block=True)
						host_controller_grpc_client.update_latest_validation_score(val=local_model_evaluation_score)

					# Synchronous call, we send the new variables to controller.
					host_controller_grpc_client.send_model_local_trained_variables_to_controller(block=True)

					# Inform Controller that the learner just finished training.
					host_controller_grpc_client.notify_controller_for_fedround_signals(finished_training=True,
																					   finished_epoch=False, block=True)

					# Add host executed result in the results HashMap.
					# TODO This is a HUGE parameterized ITEM! Better define a class for it or update the object whilst
					#  training.
					host_metadata_result = FedHostExecutionResultsStats(host_id=host_id,
						training_devices=host_training_devices, completed_epochs=host_completed_epochs,
						completed_batches=host_completed_batches, num_training_examples=training_dataset_size,
						num_validation_examples=validation_dataset_size, compute_init_unix_time=compute_init_time,
						compute_end_unix_time=TimeUtil.current_milli_time(),
						test_set_evaluations=temporal_test_evaluations,
						train_set_evaluations=temporal_train_evaluations,
						validation_set_evaluations=temporal_validation_evaluations, epochs_exec_times_ms=epochs_times,
						metis_grpc_client_metadata=host_controller_grpc_client.toJSON_representation())

				# TODO extend with grpc client metadata
				# Update federation round execution HashMap.
				hosts_execution_metadata_map[host_id] = host_metadata_result
				host_controller_grpc_client.shutdown()

				# Store the latest community model
				# model_output_file = '/tmp/metis_project/execution_logs/federation_models/learner_model_{}_federation_round_{}.npz'.format(lidx, frid)
				# np.savez(model_output_file, *final_trainable_vars)

				# Federation round training completion.
				return


	@classmethod
	def __semisync_fedhost_training(cls,
									FEDHOST_NUMBER_OF_BATCHES,
									fed_host_obj,
									federated_variables,
									federation_model_obj,
									community_function,
									metis_db_session,
									controller_host_port,
									evaluator_host_port,
									hosts_execution_metadata_map,
									session_target_stat_name,
									batch_level_log=False):
		"""
		:param fed_host_obj
		:param federated_variables:
		:param federation_model_obj:
		:param community_function:
		:param metis_db_session:
		:param controller_host_port:
		:param evaluator_host_port
		:param hosts_execution_metadata_map:
		:param session_target_stat_name
		:return:
		"""

		if not isinstance(fed_host_obj, fedenv.FedHost):
			raise TypeError("The provided fed_host_obj parameter must be of type %s" % fedenv.FedHost)
		if not any(isinstance(fedvar, fedmodel.FedVar) for fedvar in federated_variables):
			raise TypeError("All the federated variables passed to this function must be of type %s " % fedmodel.FedVar)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s" % fedmodel.FedModelDef)

		tf_cluster_spec = fed_host_obj.cluster_spec
		tf_cluster_master = fed_host_obj.fed_master
		tf_worker_server = fed_host_obj.fed_worker_servers[0]
		w_device_name = tf_worker_server.device_name
		w_is_remote_server = tf_worker_server.is_remote_server
		w_is_chief = tf_worker_server.is_leader
		w_gpu_id = tf_worker_server.gpu_id
		w_local_batch_size = tf_worker_server.batch_size
		w_local_epochs = tf_worker_server.target_update_epochs
		host_id = fed_host_obj.host_identifier
		host_training_devices = fed_host_obj.host_training_devices
		w_is_fast_learner = True if "GPU" in host_training_devices else False
		w_config = TFConfiguration.tf_session_config(is_gpu=w_is_fast_learner)

		if w_is_remote_server:
			w_device = tf.train.replica_device_setter(cluster=tf_cluster_spec, worker_device=w_device_name)
		else:
			w_device = '/gpu:0' if w_gpu_id is None else '/gpu:{}'.format(w_gpu_id)

		compute_init_time = TimeUtil.current_milli_time()
		temporal_test_evaluations = list()
		temporal_train_evaluations = list()
		temporal_validation_evaluations = list()

		# Separate the 'DVW' community weighting scheme/function from other functions (e.g., 'FedAvg').
		dvw_weighting_scheme = True if "DVW" in community_function else False
		host_completed_epochs = 0
		host_completed_batches = 0
		epochs_times = list()
		epochs_proc_times = list()


		host_controller_grpc_client = FedClient(client_id=host_id, controller_host_port=controller_host_port)

		# TODO The evaluation request needs to be moved to the controller end.
		host_evaluator_grpc_client = FedModelEvaluatorClient(client_id=host_id, evaluator_host_port=evaluator_host_port)

		with tf.device(w_device):
			tf.reset_default_graph()
			exec_graph = tf.Graph()
			with exec_graph.as_default():

				# To ensure randomly deterministic operations and values, we set the random seed.
				np.random.seed(seed=1990)
				random.seed(1990)
				tf.set_random_seed(seed=1990)

				# Import all learner datasets structure.
				train_dataset_structure, validation_dataset_structure, test_dataset_structure = metis_db_session\
					.import_host_data(learner_id=host_id,
									  batch_size=w_local_batch_size,
									  import_train=True,
									  import_validation=True,
									  import_test=True)

				training_init_op = train_dataset_structure.dataset_init_op
				next_train_dataset = train_dataset_structure.dataset_next
				training_dataset_size = train_dataset_structure.dataset_size

				validation_init_op = validation_dataset_structure.dataset_init_op
				next_validation_dataset = validation_dataset_structure.dataset_next
				validation_dataset_size = validation_dataset_structure.dataset_size

				testing_init_op = test_dataset_structure.dataset_init_op
				next_test_dataset = test_dataset_structure.dataset_next
				testing_dataset_size = test_dataset_structure.dataset_size

				# Inform GRPC client regarding the number of training examples for this learner.
				host_controller_grpc_client.update_num_training_examples(val=training_dataset_size)
				host_controller_grpc_client.update_num_validation_examples(val=validation_dataset_size)
				host_controller_grpc_client.update_batch_size(val=w_local_batch_size)

				# Retrieve model's input & output placeholders.
				_x_placeholders = federation_model_obj.input_tensors_datatype()
				_y_placeholders = federation_model_obj.output_tensors_datatype()

				# Define model's global step.
				_global_step = tf.train.get_or_create_global_step()

				# Define Deep NN Graph.
				model_architecture = federation_model_obj.model_architecture(input_tensors=_x_placeholders,
																			 output_tensors=_y_placeholders,
																			 global_step=_global_step,
																			 batch_size=w_local_batch_size,
																			 dataset_size=training_dataset_size)

				# Retrieve FedOperation and FedTensor collections, see projectmetis/federation/fed_model.py.
				train_step = model_architecture.train_step
				loss_tensor_fedmodel = model_architecture.loss
				loss_tensor = loss_tensor_fedmodel.get_tensor()
				predictions_tensor_fedmodel = model_architecture.predictions

				# Register tensorflow evaluation operations based on evaluation task: regression or classification.
				tf_graph_evals = TFGraphEvaluation(_x_placeholders, _y_placeholders, predictions_tensor_fedmodel,
												   is_classification=metis_db_session.is_classification,
												   is_regression=metis_db_session.is_regression,
												   num_classes=metis_db_session.num_classes,
												   negative_classes_indices=metis_db_session.negative_classes_indices,
												   is_eval_output_scalar=metis_db_session.is_eval_output_scalar)
				tf_graph_evals.assign_tfdatasets_operators(training_init_op, next_train_dataset, validation_init_op,
														   next_validation_dataset, testing_init_op, next_test_dataset)
				tf_graph_evals.register_evaluation_ops(metis_db_session.get_learner_evaluation_output_attribute(host_id),
													   w_local_batch_size)

				# Get the trainable (federated) variables collection defined in the model graph.
				trainable_variables = model_architecture.model_federated_variables

				# There could be a case that the testing variables are different form the training variables,
				# 	e.g. TRAINABLE collection vs MOVING_AVERAGE collection.
				# Based on this case, we take one additional step to define which variables are
				# the 'true' model's evaluation variables.
				udf_testing_vars = model_architecture.user_defined_testing_variables_collection
				if udf_testing_vars is not None:
					udf_testing_vars = TFConfiguration.tf_trainable_variables_config(
						graph_trainable_variables=trainable_variables,
						user_defined_variables_collection=udf_testing_vars)

				"""			
				The `fedvars_atomic` variables scope will be stored in the Parameter Server(PS) and shared across 
				Workers ~ thus they must be declared as GLOBAL_VARIABLES. Their purpose is to initialize the weights & 
				biases of every Worker's model. The 'reuse=tf.AUTO_REUSE' initializes the variables if they do not 
				exist, otherwise returns their value.
				"""
				with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
					for idx, variable in enumerate(federated_variables):
						tf.get_variable(name=variable.name, initializer=variable.value,
										collections=variable.tf_collection, trainable=variable.trainable)

				# Define operations for model weights and biases initialization with the federated variables values.
				fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
				assign_atomic_feds = []
				for (index, model_var) in enumerate(trainable_variables):
					fed_var = fed_vars[index]
					assign_atomic_feds.append(tf.assign(model_var, fed_var))

				# A verification operation to make sure that all model variables have been initialized.
				uninitialized_vars = tf.report_uninitialized_variables()

				# model_saver_hook = tf.train.CheckpointSaverHook(
				# 	checkpoint_dir='/nfs/isd/stripeli/metis_execution_models_checkpoint_dir/',
				# 	save_secs=5,
				# 	save_steps=int(np.floor(np.divide(training_dataset_size, w_local_batch_size))),
				# 	saver=tf.train.Saver(),
				# 	checkpoint_basename='{}.model.ckpt'.format("".join(c for c in host_id if c.isalnum())),
				# 	scaffold=None)

				# TODO Reduce delay in terms of workers training initialization, see below:
				""" 
				Why there is delay between workers? Because Monitored Training Session checks whether the variables 
				in the underlying graph have been initialized or not. If the variables are not initialized yet an 
				warning/info will be printed as follows and a racing between the workers will occur:
					INFO:tensorflow:Waiting for model to be ready.  Ready_for_local_init_op:  Variables not initialized: 
				Then the MonitoredTrainingSession will put the 'uninitialized' worker(s) to sleep for 30 seconds and 
				repeat. Further reading: 
				https://stackoverflow.com/questions/43084960/
					tensorflow-variables-are-not-initialized-using-between-graph-replication
				"""

				# The 'is_chief' field in the MonitoringTrainingSession indicates whether we want a Master or a Worker
				# tensorflow graph training session.
				tf_master_grpc = tf_cluster_master.grpc_endpoint
				with tf.train.MonitoredTrainingSession(master=tf_master_grpc, is_chief=w_is_chief, config=w_config) \
						as mon_sess:

					# As long as we have uninitialized variables, we wait for the TF server to initialize all.
					while len(mon_sess.run(uninitialized_vars)) > 0:
						metis_logger.info(msg="Host TF Cluster %d: waiting for variables initialization..." % host_id)
						time.sleep(1.0)

					metis_logger.info("Idle time due to Tensorflow graph operations creation (secs): {}".format(
						TimeUtil.delta_diff_in_secs(TimeUtil.current_milli_time(), compute_init_time)))

					# Assign the community model to the learner.
					mon_sess.run(assign_atomic_feds)

					single_epoch_total_batches = int(np.ceil(np.divide(training_dataset_size, w_local_batch_size)))
					if FEDHOST_NUMBER_OF_BATCHES is None:
						FEDHOST_NUMBER_OF_BATCHES = single_epoch_total_batches

					# Always perform one more iteration to satisfy the since 0 accounts for 1 batch.
					# We increase by one to satisfy the while conditions and perform the entire epoch.
					metis_logger.info("Learner: {} trains over a total of: {} batches.".format(
						host_id, FEDHOST_NUMBER_OF_BATCHES))


					# TODO HACK!!
					# Evaluate Community Model on local dataset
					train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals \
						.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

					metis_logger.info("Host ID: {}, Community Model Train Evaluation Results: {}".format(
						host_id, train_eval_results))
					metis_logger.info("Host ID: {}, Community Model Validation Evaluation Results: {}".format(
						host_id, validation_eval_results))
					metis_logger.info("Host ID: {}, Community Model Test Evaluation Results: {}".format(
						host_id, test_eval_results))

					temporal_train_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	  eval_results=train_eval_results,
																	  is_after_community_update=True))
					temporal_validation_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																		   eval_results=validation_eval_results,
																		   is_after_community_update=True))
					temporal_test_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	 eval_results=test_eval_results,
																	 is_after_community_update=True))



					# Initialize epoch sentinel variables.
					epoch_current_batch_num = 0
					epoch_start_time = TimeUtil.current_milli_time()
					epoch_process_start_time = TimeUtil.current_process_time()

					# Two loops with same condition. One for the inner dataset.
					while host_completed_batches < FEDHOST_NUMBER_OF_BATCHES:
						# Initialize training dataset iterator.
						mon_sess.run(training_init_op)

						while host_completed_batches < FEDHOST_NUMBER_OF_BATCHES:

							try:

								# Load the next training batch.
								train_batch = mon_sess.run(next_train_dataset)

								train_extra_feeds = OrderedDict()
								# We need to retrieve any additional placeholders declared during the construction of
								# the network. In this step we need to get the placeholders for the federated model
								# training operation. e.g. {'is_training': True}
								train_extra_feeds.update(train_step.get_feed_dictionary())
								for placeholder_name, placeholder_def in _x_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
								for placeholder_name, placeholder_def in _y_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

								if epoch_current_batch_num == 0:
									first_batch_process_start_time = TimeUtil.current_process_time()

								# Train model (train_step FedOperation).
								train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)

								if epoch_current_batch_num == 0:
									first_batch_process_end_time = TimeUtil.current_process_time()
									first_batch_process_duration = TimeUtil.delta_diff_in_ms(
										first_batch_process_start_time, first_batch_process_end_time)
									metis_logger.info("{}, First batch train step: {}".format(
										host_id, first_batch_process_duration))

								epoch_current_batch_num += 1
								host_completed_batches += 1
								# Update grpc client's number of completed batches.
								host_controller_grpc_client.update_completed_batches(val=host_completed_batches)

							except tf.errors.OutOfRangeError:
								break

						if single_epoch_total_batches == epoch_current_batch_num:
							# Iterator exception means that we completed an epoch.
							host_completed_epochs += 1

							# Compute epoch computation/processing time.
							epoch_end_time = TimeUtil.current_milli_time()
							epoch_duration = TimeUtil.delta_diff_in_ms(epoch_start_time, epoch_end_time)
							epochs_times.append(epoch_duration)

							epoch_process_end_time = TimeUtil.current_process_time()
							epoch_process_duration = TimeUtil.delta_diff_in_ms(epoch_process_start_time,
																			   epoch_process_end_time)
							# We remove the process time for the first batch operation since, this requires to first
							# materialize the tensorflow graph.
							amortized_process_epoch_duration = epoch_process_duration - first_batch_process_duration
							epochs_proc_times.append(amortized_process_epoch_duration)
							metis_logger.info("{}, Amortized epoch time: {}".format(
								host_id, epoch_process_duration - first_batch_process_duration))

							# Update epoch time value of the grpc controller client.
							host_controller_grpc_client.update_processing_ms_per_epoch(val=np.mean(epochs_times))

							# Update controller that the learner just finished a training epoch.
							host_controller_grpc_client.notify_controller_for_fedround_signals(finished_training=False,
																							   finished_epoch=True,
																							   block=True)

							# Since the variables we need to use to test our model could be the ones defined by the user
							# (e.g. MovingAverage variables) we need compute the final variables values and then re-assign
							# them to the network trainable variables.
							final_trainable_vars = mon_sess.run(trainable_variables)

							# Update grpc client's variables with trained variables.
							host_controller_grpc_client.update_trained_variables(new_variables=final_trainable_vars)

							# Restart epoch batch and time counters. More objective values if defined here right before
							# starting the next epoch!
							epoch_current_batch_num = 0
							epoch_start_time = TimeUtil.current_milli_time()
							epoch_process_start_time = TimeUtil.current_process_time()

					# In order to speed up training we can evaluate
					# If we have different testing variables from trainable variables, then
					# update the model with the testing variables and proceed with model evaluation
					if udf_testing_vars is not None:
						final_testing_vars = mon_sess.run(udf_testing_vars)
						for index, final_var_value in enumerate(final_testing_vars):
							fed_vars[index].load(final_var_value, mon_sess)
						mon_sess.run(assign_atomic_feds)

					train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals\
						.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

					metis_logger.info("Host ID: {}, Local Model Train Evaluation Results: {}".format(
						host_id, train_eval_results))
					metis_logger.info("Host ID: {}, Local Model Validation Evaluation Results: {}".format(
						host_id, validation_eval_results))
					metis_logger.info("Host ID: {}, Local Model Test Evaluation Results: {}".format(
						host_id, test_eval_results))

					temporal_train_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	  eval_results=train_eval_results,
																	  is_after_epoch_completion=True))
					temporal_validation_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																		   eval_results=validation_eval_results,
																		   is_after_epoch_completion=True))
					temporal_test_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																	 eval_results=test_eval_results,
																	 is_after_epoch_completion=True))

					latest_train_accuracy = train_eval_results[session_target_stat_name]
					host_controller_grpc_client.update_latest_train_score(val=latest_train_accuracy)
					latest_validation_score = validation_eval_results[session_target_stat_name]
					host_controller_grpc_client.update_latest_validation_score(val=latest_validation_score)

					# Get the latest trainable variables for evaluation.
					final_trainable_vars = mon_sess.run(trainable_variables)

					# If the testing variables are defined then the network holds these testing variables, and now
					# we need to assign the trainable variables in order to proceed with the training.
					if udf_testing_vars is not None:
						for index, final_var_value in enumerate(final_trainable_vars):
							fed_vars[index].load(final_var_value, mon_sess)
							federated_variables[index].value = final_var_value
						mon_sess.run(assign_atomic_feds)

					# Update grpc client's variables with trained variables.
					host_controller_grpc_client.update_trained_variables(new_variables=final_trainable_vars)

					# End of local training. If DVW weighting scheme, then evaluate local model.
					if dvw_weighting_scheme:
						local_model_evaluation_score = host_evaluator_grpc_client.request_model_evaluation(
							model_variables=final_trainable_vars, is_community_model=False, block=True)
						host_controller_grpc_client.update_latest_validation_score(val=local_model_evaluation_score)

					# Synchronous call, we send the new variables to controller.
					host_controller_grpc_client.send_model_local_trained_variables_to_controller(block=True)

					# Inform Controller that the learner just finished training.
					host_controller_grpc_client.notify_controller_for_fedround_signals(finished_training=True,
																					   finished_epoch=False, block=True)

					# Add host executed result in the results HashMap.
					# TODO This is a HUGE parameterized ITEM! Better define a class for it or update the object whilst
					#  training.
					host_metadata_result = FedHostExecutionResultsStats(host_id=host_id,
						training_devices=host_training_devices, completed_epochs=host_completed_epochs,
						completed_batches=host_completed_batches, num_training_examples=training_dataset_size,
						num_validation_examples=validation_dataset_size, compute_init_unix_time=compute_init_time,
						compute_end_unix_time=TimeUtil.current_milli_time(),
						test_set_evaluations=temporal_test_evaluations,
						train_set_evaluations=temporal_train_evaluations,
						validation_set_evaluations=temporal_validation_evaluations, epochs_exec_times_ms=epochs_times,
						epochs_proc_times_ms=epochs_proc_times,
						metis_grpc_client_metadata=host_controller_grpc_client.toJSON_representation())

				# TODO extend with grpc client metadata
				# Update federation round execution HashMap.
				hosts_execution_metadata_map[host_id] = host_metadata_result
				host_controller_grpc_client.shutdown()

				# Federation round training completion.
				return


	@classmethod
	def __async_fedhost_training(cls,
								 fed_host_obj,
								 federated_variables,
								 federation_model_obj,
								 community_function,
								 metis_db_session,
								 controller_host_port,
								 evaluator_host_port,
								 hosts_execution_metadata_map,
								 session_target_stat_name,
								 batch_level_log=False):
		"""
		:param fed_host_obj
		:param federated_variables:
		:param federation_model_obj:
		:param metis_db_session:
		:param controller_host_port:
		:param hosts_execution_metadata_map:
		:param session_target_stat_name:
		:param batch_level_log:
		:return:
		"""

		if not isinstance(fed_host_obj, fedenv.FedHost):
			raise TypeError("The provided fed_host_obj parameter must be of type %s" % fedenv.FedHost)
		if not any(isinstance(fedvar, fedmodel.FedVar) for fedvar in federated_variables):
			raise TypeError("All the federated variables passed to this function must be of type %s " % fedmodel.FedVar)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s" % fedmodel.FedModelDef)

		tf_cluster_spec = fed_host_obj.cluster_spec
		tf_cluster_master = fed_host_obj.fed_master
		tf_worker_server = fed_host_obj.fed_worker_servers[0]
		w_device_name = tf_worker_server.device_name
		w_is_remote_server = tf_worker_server.is_remote_server
		w_is_chief = tf_worker_server.is_leader
		w_gpu_id = tf_worker_server.gpu_id
		w_local_batch_size = tf_worker_server.batch_size
		w_local_epochs = tf_worker_server.target_update_epochs
		w_config = TFConfiguration.tf_session_config(gpu_id=w_gpu_id)
		host_id = fed_host_obj.host_identifier
		host_training_devices = fed_host_obj.host_training_devices
		w_is_fast_learner = True if "GPU" in host_training_devices else False

		if w_is_remote_server:
			w_device = tf.train.replica_device_setter(cluster=tf_cluster_spec, worker_device=w_device_name)
		else:
			w_device = '/gpu:0' if w_gpu_id is None else '/gpu:{}'.format(w_gpu_id)

		compute_init_time = TimeUtil.current_milli_time()
		temporal_test_evaluations = list()
		temporal_train_evaluations = list()
		temporal_validation_evaluations = list()

		# Separate the 'DVW' community weighting scheme/function from other functions (e.g., 'FedAvg').
		dvw_weighting_scheme = True if "DVW" in community_function else False
		host_completed_epochs = 0
		host_completed_batches = 0
		epochs_times = list()
		current_update_request_num = 0
		extra_measurements = defaultdict(list)
		current_global_epoch_id = 1
		global_epoch_transition_threshold = 1000000
		community_model_validation_loss = list()

		# All variables below refer to the asynchronous execution.
		__burnin_epochs = 0
		__target_burnin_epochs = 2
		__learner_in_validation_phase = True if w_local_epochs == 0 else False

		# Following sentinel variables are set for validation cycles (VCs).
		__vc_target_validation_loss_tombstones = fed_host_obj.fed_master.validation_cycle_tombstones
		__vc_target_validation_loss_change = fed_host_obj.fed_master.validation_cycle_loss_percentage_threshold
		__vc_optimal_validation_loss = math.inf
		__vc_first_recorded_training_loss = None
		__vc_first_recorded_validation_loss = None

		# The following variables need to be reset upon a validation cycle completion.
		__vc_train_loss_window = list()
		__vc_train_loss_window_length = 1

		__vc_validation_loss_window = list()
		__vc_validation_loss_window_length = 1

		__vc_local_epochs = 0
		__vc_completed_batches = 0
		__vcs_local_epochs = list()
		__vcs_staleness = list()
		__vcs_weighting_values = defaultdict(list)
		__vc_learner_tombstone = 0
		__are_stopping_criteria_reached = False

		# Each learner/grpc_client in the asynchronous execution listens for session termination signals/shutdown
		host_controller_grpc_client = FedClient(client_id=host_id, controller_host_port=controller_host_port,
												listen_for_session_termination_signals=True)

		# TODO This needs to be moved to the controller end. It is okay for now.
		host_evaluator_grpc_client = FedModelEvaluatorClient(client_id=host_id, evaluator_host_port=evaluator_host_port)

		with tf.device(w_device):
			tf.reset_default_graph()
			exec_graph = tf.Graph()
			with exec_graph.as_default():

				# To ensure deterministic operations and random values, helpful for debugging,
				# we need to set the random seed
				random.seed(1990)
				tf.set_random_seed(seed=1990)
				np.random.seed(seed=1990)

				# Import all learner datasets structure.
				train_dataset_structure, validation_dataset_structure, test_dataset_structure = metis_db_session\
					.import_host_data(learner_id=host_id,
									  batch_size=w_local_batch_size,
									  import_train=True,
									  import_validation=True,
									  import_test=True)

				training_init_op = train_dataset_structure.dataset_init_op
				next_train_dataset = train_dataset_structure.dataset_next
				training_dataset_size = train_dataset_structure.dataset_size

				validation_init_op = validation_dataset_structure.dataset_init_op
				next_validation_dataset = validation_dataset_structure.dataset_next
				validation_dataset_size = validation_dataset_structure.dataset_size

				testing_init_op = test_dataset_structure.dataset_init_op
				next_test_dataset = test_dataset_structure.dataset_next
				testing_dataset_size = test_dataset_structure.dataset_size

				# Inform GRPC client regarding the number of training and validation examples for this learner.
				host_controller_grpc_client.update_num_training_examples(val=training_dataset_size)
				host_controller_grpc_client.update_num_validation_examples(val=validation_dataset_size)
				host_controller_grpc_client.update_batch_size(val=w_local_batch_size)

				# Define Model's input & output format.
				_x_placeholders = federation_model_obj.input_tensors_datatype()
				_y_placeholders = federation_model_obj.output_tensors_datatype()

				# Define model's global step, size of the host's local batch and dataset.
				_global_step = tf.train.get_or_create_global_step()

				# Define Deep NN Graph.
				model_architecture = federation_model_obj.model_architecture(input_tensors=_x_placeholders,
																			 output_tensors=_y_placeholders,
																			 global_step=_global_step,
																			 batch_size=w_local_batch_size,
																			 dataset_size=training_dataset_size)

				# Retrieve FedOperation and FedTensor collections, see projectmetis/federation/fed_model.py.
				train_step = model_architecture.train_step
				loss_tensor_fedmodel = model_architecture.loss
				loss_tensor = loss_tensor_fedmodel.get_tensor()
				predictions_tensor_fedmodel = model_architecture.predictions

				# Register tensorflow evaluation operations based on evaluation task: regression or classification.
				tf_graph_evals = TFGraphEvaluation(_x_placeholders, _y_placeholders, predictions_tensor_fedmodel,
												   is_classification=metis_db_session.is_classification,
												   is_regression=metis_db_session.is_regression,
												   num_classes=metis_db_session.num_classes,
												   negative_classes_indices=metis_db_session.negative_classes_indices,
												   is_eval_output_scalar=metis_db_session.is_eval_output_scalar)
				tf_graph_evals.assign_tfdatasets_operators(training_init_op, next_train_dataset, validation_init_op,
														   next_validation_dataset, testing_init_op, next_test_dataset)
				tf_graph_evals.register_evaluation_ops(metis_db_session.get_learner_evaluation_output_attribute(host_id),
													   w_local_batch_size)


				# Get trainable variables collection defined in the model graph.
				trainable_variables = model_architecture.model_federated_variables

				# There could be a case that the testing variables are different form the training variables,
				# 	e.g. TRAINABLE collection vs MOVING_AVERAGE collection.
				# Based on this case, we take one additional step to define which variables are
				# the 'true' model's evaluation variables.
				udf_testing_vars = model_architecture.user_defined_testing_variables_collection
				if udf_testing_vars is not None:
					udf_testing_vars = TFConfiguration.tf_trainable_variables_config(
						graph_trainable_variables=trainable_variables,
						user_defined_variables_collection=udf_testing_vars)

				"""			
				The `fedvars_atomic` variables scope will be stored in the Parameter Server(PS) and shared across 
				Workers ~ thus they must be declared as GLOBAL_VARIABLES. Their purpose is to initialize the weights & 
				biases of every Worker's model. The 'reuse=tf.AUTO_REUSE' initializes the variables if they do not 
				exist, otherwise returns their value.
				"""
				with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
					for variable in federated_variables:
						tf.get_variable(name=variable.name, initializer=variable.value,
										collections=variable.tf_collection, trainable=variable.trainable)

				# Define operations for model weights and biases initialization with the federated variables values.
				fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
				assign_atomic_feds = []
				for (index, model_var) in enumerate(trainable_variables):
					fed_var = fed_vars[index]
					assign_atomic_feds.append(tf.assign(model_var, fed_var))

				# A verification operation to make sure that all model variables have been initialized.
				uninitialized_vars = tf.report_uninitialized_variables()

				# TODO Reduce delay in terms of workers training initialization, see below
				""" 
				Why there is delay between workers? Because Monitored Training Session checks whether the variables 
				in the underlying graph have been initialized or not. If the variables are not initialized yet an 
				warning/info will be printed as follows and a racing between the workers will occur:
					INFO:tensorflow:Waiting for model to be ready.  Ready_for_local_init_op:  Variables not initialized: 
				Then the MonitoredTrainingSession will put the 'uninitialized' worker(s) to sleep for 30 seconds and 
				repeat. Further reading: 
				https://stackoverflow.com/questions/43084960/
					tensorflow-variables-are-not-initialized-using-between-graph-replication
				"""

				# The 'is_chief' field in the MonitoringTrainingSession indicates whether we want a Master or a Worker
				# tensorflow graph training session.
				tf_master_grpc = tf_cluster_master.grpc_endpoint
				with tf.train.MonitoredTrainingSession(master=tf_master_grpc, is_chief=w_is_chief, config=w_config) \
						as mon_sess:

					# As long as we have uninitialized variables, we wait for the TF server to initialize all.
					while len(mon_sess.run(uninitialized_vars)) > 0:
						metis_logger.info(msg="Host TF Cluster %d: waiting for variables initialization..." % host_id)
						time.sleep(1.0)

					metis_logger.info("Idle time due to Tensorflow graph operations creation (secs): {}".format(
						TimeUtil.delta_diff_in_secs(TimeUtil.current_milli_time(), compute_init_time)))

					# Assign the community model to the learner.
					mon_sess.run(assign_atomic_feds)

					# Start asynchronous training.
					while True:

						mon_sess.run(training_init_op)
						epoch_current_batch_num = 0
						epoch_start_time = TimeUtil.current_milli_time()

						global_update_scalar_clock, learner_global_update_scalar_clock = \
							host_controller_grpc_client.request_community_and_learner_global_scalar_clock()
						community_update_requests_staleness = global_update_scalar_clock + \
															  learner_global_update_scalar_clock + 1
						global_community_steps, learner_previous_global_community_steps = \
							host_controller_grpc_client.request_community_and_learner_global_community_steps()
						community_update_steps_staleness = \
							global_community_steps - learner_previous_global_community_steps + __vc_completed_batches

						if community_update_steps_staleness > 1:
							fed_annealing_value = np.divide(1, np.sqrt(community_update_steps_staleness))
						else:
							fed_annealing_value = 0
						previous_global_epoch_id = current_global_epoch_id
						current_global_epoch_id = host_controller_grpc_client.request_current_global_epoch_id()
						metis_logger.info('Host: %s, Training Devices: %s, Global Epoch id: %s, Global Community Steps '
										  '%s,  Learner Previous Community Steps: %s, Annealing Factor: %s'
										  % (host_id, host_training_devices, current_global_epoch_id,
											 global_community_steps, learner_previous_global_community_steps +
											 __vc_completed_batches, fed_annealing_value))

						# Single epoch training.
						while True:

							try:

								# Code block that grabs the community update and applies it to the model.
								if host_controller_grpc_client.community_update_received:

									# Restart Momentum. Set to zero.
									# if current_global_epoch_id != previous_global_epoch_id:
									# 	mon_sess.run([var.initializer for var in tf.global_variables() if 'Momentum' in
									# 				  var.name])
									# Restart internal state of the tensorflow computational graph
									# mon_sess.run(reset_tf_graph_op)

									# Re-initialize the iterator when a community update occurs.
									# Start training from scratch.
									mon_sess.run(training_init_op)
									epoch_current_batch_num = 0

									metis_logger.info('Host: %s, Training Devices: %s, Received community update' %
													  (host_id, host_training_devices))
									community_update_vars = host_controller_grpc_client\
										.retrieve_latest_community_update_with_flag_toggle()

									for index, community_var_value in enumerate(community_update_vars):
										fed_vars[index].load(community_var_value, mon_sess)
										federated_variables[index].value = community_var_value
									mon_sess.run(assign_atomic_feds)
									metis_logger.info('Host: %s, Training Devices: %s, Community update applied on the '
													  'network' % (host_id, host_training_devices))

									train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals \
										.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

									metis_logger.info("Host ID: {}, Community Model Train Evaluation Results: {}"
										.format(host_id, train_eval_results))
									metis_logger.info("Host ID: {}, Community Model Validation Evaluation Results: {}"
										.format(host_id, validation_eval_results))
									metis_logger.info("Host ID: {}, Community Model Test Evaluation Results: {}"
										.format(host_id, test_eval_results))

									temporal_train_evaluations.append(
										FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
														eval_results=train_eval_results,
														is_after_community_update=True))

									temporal_validation_evaluations.append(
										FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
														eval_results=validation_eval_results,
														is_after_community_update=True))
									community_model_validation_loss.append(
										validation_eval_results['validation_loss_mean'])

									temporal_test_evaluations.append(
										FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
														eval_results=test_eval_results,
														is_after_community_update=True))

								# Code block to check whether any termination signal is reached before a new batch.
								if host_controller_grpc_client.is_session_termination_signal_reached:

									metis_logger.info(msg="Host: {}, Federation round terminated, wrapping up "
														  "computation".format(host_id))

									# Capture host metadata.
									host_metadata_result = FedHostExecutionResultsStats(
										host_id=host_id, training_devices=host_training_devices,
										completed_epochs=host_completed_epochs,
										completed_batches=host_completed_batches,
										num_training_examples=training_dataset_size,
										num_validation_examples=validation_dataset_size,
										compute_init_unix_time=compute_init_time,
										compute_end_unix_time=TimeUtil.current_milli_time(),
										test_set_evaluations=temporal_test_evaluations,
										train_set_evaluations=temporal_train_evaluations,
										validation_set_evaluations=temporal_validation_evaluations,
										epochs_exec_times_ms=epochs_times,
										metis_grpc_client_metadata=host_controller_grpc_client
											.toJSON_representation())

									# Request Current Community
									current_community_model = host_controller_grpc_client\
										.retrieve_latest_community_update()
									for idx in range(len(federated_variables)):
										federated_variables[idx].value = current_community_model[idx]

									# TODO extend with grpc client metadata
									# Update Execution HashMap
									hosts_execution_metadata_map[host_id] = host_metadata_result
									host_controller_grpc_client.shutdown()
									host_evaluator_grpc_client.shutdown()

									# End of Asynchronous training.
									return

								# Load the next training batch.
								train_batch = mon_sess.run(next_train_dataset)
								train_extra_feeds = OrderedDict()
								train_extra_feeds.update(train_step.get_feed_dictionary())
								for placeholder_name, placeholder_def in _x_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
								for placeholder_name, placeholder_def in _y_placeholders.items():
									train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

								# Code block on how to manipulate learning rate
								# with_lr_federated_annealing = int(os.environ["WITH_LR_FEDERATED_ANNEALING"])
								# if with_lr_federated_annealing == 1 and \
								# 		current_global_epoch_id > global_epoch_transition_threshold:
								# 	train_extra_feeds['lr_annealing_value:0'] = fed_annealing_value
								# else:
								# 	train_extra_feeds['lr_annealing_value:0'] = 0
								# with_momentum_federated_annealing = int(os.environ["WITH_MOMENTUM_FEDERATED_ANNEALING"])
								# if with_momentum_federated_annealing == 1 and \
								# 		current_global_epoch_id > global_epoch_transition_threshold:
								# 	train_extra_feeds['momentum_annealing_value:0'] = fed_annealing_value
								# else:
								# 	train_extra_feeds['momentum_annealing_value:0'] = 0

								# lr_value = mon_sess.run("lr_value:0", feed_dict=train_extra_feeds)
								# momentum_value = mon_sess.run("momentum_value:0", feed_dict=train_extra_feeds)
								# metis_logger.info(msg="LR: {}, Momentum: {}".format(lr_value, momentum_value))

								# Train model (train_step FedOperation).
								train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)
								epoch_current_batch_num += 1
								host_completed_batches += 1
								host_controller_grpc_client.update_completed_batches(val=host_completed_batches)

								if __learner_in_validation_phase:
									__vc_completed_batches += 1

								# If logging is set, print every 10 batches.
								if batch_level_log and epoch_current_batch_num % 10 == 0:

									# Fetch the current global step.
									current_step = mon_sess.run(_global_step)
									if loss_tensor is not None:
										train_extra_feeds.update(predictions_tensor_fedmodel.get_feed_dictionary())
										train_loss = loss_tensor.eval_tf_tensor(session=mon_sess,
																				extra_feeds=train_extra_feeds)
										metis_logger.info(msg='Host: %s, Training Devices: %s, epoch %d, batch %d, '
															  'global step %d, loss %g'
															  % (host_id, host_training_devices, host_completed_epochs+1,
																 epoch_current_batch_num, current_step, train_loss))

							except tf.errors.OutOfRangeError:
								break

						# Code block to use the validation data into training.
						# If we have reached the update requests cutoff point (i.e. learner is no more in adaptive
						# execution mode) then for every epoch we perform a training pass over the network using the
						# validation dataset
						# if learner_in_validation_phase is False:
						# 	mon_sess.run(validation_init_op)
						# 	validation_current_batch_num = 0
						# 	while True:
						# 		try:
						#
						# 			validation_batch = mon_sess.run(next_validation_dataset)
						# 			validation_train_extra_feeds = OrderedDict()
						# 			for placeholder_name, placeholder_def in _x_placeholders.items():
						# 				validation_train_extra_feeds[placeholder_def] = validation_batch[placeholder_name]
						# 			for placeholder_name, placeholder_def in _y_placeholder.items():
						# 				validation_train_extra_feeds[placeholder_def] = validation_batch[placeholder_name]
						#
						# 			with_lr_federated_annealing = int(os.environ["WITH_LR_FEDERATED_ANNEALING"])
						# 			if with_lr_federated_annealing == 1 and current_global_epoch_id > global_epoch_transition_threshold:
						# 				validation_train_extra_feeds['lr_annealing_value:0'] = fed_annealing_value
						# 			else:
						# 				validation_train_extra_feeds['lr_annealing_value:0'] = 0
						# 			with_momentum_federated_annealing = int(os.environ["WITH_MOMENTUM_FEDERATED_ANNEALING"])
						# 			if with_momentum_federated_annealing == 1 and current_global_epoch_id > global_epoch_transition_threshold:
						# 				validation_train_extra_feeds['momentum_annealing_value:0'] = fed_annealing_value
						# 			else:
						# 				validation_train_extra_feeds['momentum_annealing_value:0'] = 0
						#
						# 			# Train Model on Validation Batch
						# 			train_step.run_tf_operation(session=mon_sess, extra_feeds=validation_train_extra_feeds)
						# 			validation_current_batch_num += 1
						# 			epoch_current_batch_num += 1
						#
						# 			# Update host local batch completed counter
						# 			host_completed_batches += 1
						# 			vc_completed_batches += 1
						#
						# 		except tf.errors.OutOfRangeError:
						# 			break

						epoch_end_time = TimeUtil.current_milli_time()

						# Compute Training Wall Clock Time in Milliseconds and Store Value
						epochs_times.append(TimeUtil.delta_diff_in_ms(epoch_start_time, epoch_end_time))

						# Update host local epoch counter
						host_completed_epochs += 1
						__vc_local_epochs += 1
						host_controller_grpc_client.update_completed_epochs(val=host_completed_epochs)
						host_controller_grpc_client.update_processing_ms_per_epoch(val=np.mean(epochs_times))

						# Update Controller that the current learner finished an epoch
						host_controller_grpc_client.notify_controller_for_fedround_signals(finished_training=False,
																						   finished_epoch=True,
																						   block=True)

						# Update client variables with trained variables
						final_trainable_vars = mon_sess.run(trainable_variables)

						# If we have different testing variables from trainable variables, then:
						# Update the model with the testing variables and proceed with model evaluation
						if udf_testing_vars is not None:
							final_testing_vars = mon_sess.run(udf_testing_vars)
							for index, final_var_value in enumerate(final_testing_vars):
								fed_vars[index].load(final_var_value, mon_sess)
							mon_sess.run(assign_atomic_feds)

						train_eval_results, validation_eval_results, test_eval_results = tf_graph_evals \
							.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id)

						metis_logger.info("Host ID: {}, Local Model Train Evaluation Results: {}"
										  .format(host_id, train_eval_results))
						metis_logger.info("Host ID: {}, Local Model Validation Evaluation Results: {}"
										  .format(host_id, validation_eval_results))
						metis_logger.info("Host ID: {}, Local Model Test Evaluation Results: {}"
										  .format(host_id, test_eval_results))

						temporal_train_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																		  eval_results=train_eval_results,
																		  is_after_epoch_completion=True))
						temporal_validation_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																			   eval_results=validation_eval_results,
																			   is_after_epoch_completion=True))
						temporal_test_evaluations.append(FedHostTempEval(unix_ms=TimeUtil.current_milli_time(),
																		 eval_results=test_eval_results,
																		 is_after_epoch_completion=True))

						# Capture first recorded value of the cycle for normalization purposes.
						if __learner_in_validation_phase and __vc_first_recorded_training_loss is None \
								and __vc_first_recorded_validation_loss is None:
							vc_first_recorded_training_loss = train_eval_results['train_loss_mean']
							vc_first_recorded_validation_loss = validation_eval_results['validation_loss_mean']

						# Collect train set measurements.
						latest_train_accuracy = train_eval_results[session_target_stat_name]
						latest_train_loss_mean = train_eval_results['train_loss_mean']
						latest_train_loss_variance = train_eval_results['train_loss_variance']
						latest_train_loss_std = np.sqrt(latest_train_loss_variance)
						host_controller_grpc_client.update_latest_train_score(val=latest_train_loss_mean)

						# Collect validation set measurements.
						latest_validation_accuracy = validation_eval_results[session_target_stat_name]
						latest_validation_loss_mean = validation_eval_results['validation_loss_mean']
						latest_validation_loss_variance = validation_eval_results['validation_loss_variance']
						latest_validation_loss_std = np.sqrt(latest_validation_loss_variance)

						# Check Validation Loss Window to decide whether to issue a request or not
						if __learner_in_validation_phase and __are_stopping_criteria_reached is False:

							latest_train_loss_mean_normalized = np.divide(latest_train_loss_mean,
																		  __vc_first_recorded_training_loss)
							latest_validation_loss_mean_normalized = np.divide(latest_validation_loss_mean,
																			   __vc_first_recorded_validation_loss)

							# Add train loss and validation loss to the sliding window
							if len(__vc_train_loss_window) < __vc_train_loss_window_length and \
									len(__vc_validation_loss_window) < __vc_validation_loss_window_length:
								__vc_train_loss_window.append(latest_train_loss_mean_normalized)
								__vc_validation_loss_window.append(latest_validation_loss_mean_normalized)
							else:
								previous_train_loss_mvg_avg = np.divide(np.sum(__vc_train_loss_window),
																		len(__vc_train_loss_window))
								__vc_train_loss_window.pop(0)
								__vc_train_loss_window.append(latest_train_loss_mean_normalized)
								current_train_loss_mvg_avg = np.divide(np.sum(__vc_train_loss_window),
																	   len(__vc_train_loss_window))
								train_loss_pct_change = 100 * np.divide(
									np.abs(current_train_loss_mvg_avg - previous_train_loss_mvg_avg),
									previous_train_loss_mvg_avg)
								metis_logger.info(msg='%s, Local Train Set Loss Percentage Change: %s' %
													  (host_id, train_loss_pct_change))

								previous_validation_loss_mvg_avg = np.divide(np.sum(__vc_validation_loss_window),
																			 len(__vc_validation_loss_window))
								__vc_validation_loss_window.pop(0)
								__vc_validation_loss_window.append(latest_validation_loss_mean_normalized)
								current_validation_loss_mvg_avg = np.divide(np.sum(__vc_validation_loss_window),
																			len(__vc_validation_loss_window))
								validation_loss_pct_change = 100 * np.divide(
									np.abs(previous_validation_loss_mvg_avg - current_validation_loss_mvg_avg),
									previous_validation_loss_mvg_avg)
								metis_logger.info(msg='%s, Local Validation Set Loss Percentage Change: %s' %
													  (host_id, validation_loss_pct_change))

								# Keep track of the optimal validation loss within the VC up to this point
								if latest_validation_loss_mean < __vc_optimal_validation_loss:
									__vc_optimal_validation_loss = latest_validation_loss_mean
									__vc_optimal_weights = final_trainable_vars

								""" THE SLOPE LOSS CRITERION """
								validation_loss_slope_steepness = np.subtract(current_validation_loss_mvg_avg,
																			  previous_validation_loss_mvg_avg)
								metis_logger.info(msg='%s, Local Validation Set Loss Slope: %s' %
													  (host_id, validation_loss_slope_steepness))
								# if positive slope then increase tombstone (learner did not learn)
								if validation_loss_slope_steepness > 0:
									__vc_learner_tombstone += 1
								else:
									# Check steepness of the slope.
									# if negative slope, then check if absolute value of negative slope
									# is less than the threshold. Thus increase tombstone
									# e.g. if |-0.001| < 0.002 increase tombstone
									# abs_validation_loss_slope_steepness = np.abs(validation_loss_slope_steepness)
									# if abs_validation_loss_slope_steepness < vc_target_validation_loss_change:
									# 	vc_learner_tombstone += 1
									# Check the percentage change in validation loss
									if validation_loss_pct_change < __vc_target_validation_loss_change:
										__vc_learner_tombstone += 1
								if __vc_learner_tombstone > __vc_target_validation_loss_tombstones:
									are_stopping_criteria_reached = True

								""" THE STALENESS AND UPDATE FREQUENCY CRITERION """
								global_community_steps, learner_previous_global_community_steps = \
									host_controller_grpc_client.request_community_and_learner_global_community_steps()
								community_update_steps_staleness = \
									global_community_steps - learner_previous_global_community_steps + __vc_completed_batches

								if len(__vcs_local_epochs) > 21 and len(__vcs_staleness) > 21:
									validation_phase_local_epochs = __vcs_local_epochs[2:22]  # exclude first two attempts
									validation_phase_stalenesses = __vcs_staleness[2:22]  # exclude first two attempts

									historical_staleness_mode = stats.mode(validation_phase_stalenesses)[0][0]
									historical_staleness_mean = np.mean(validation_phase_stalenesses)
									historical_staleness_median = np.median(validation_phase_stalenesses)
									historical_staleness_std = np.std(validation_phase_stalenesses)

									historical_local_epochs_mode = stats.mode(validation_phase_local_epochs)[0][0]
									historical_local_epochs_mean = np.mean(validation_phase_local_epochs)
									historical_local_epochs_median = np.median(validation_phase_local_epochs)
									historical_local_epochs_std = np.std(validation_phase_local_epochs)

									# vc_local_epochs_annealing_factor = (current_global_epoch_id-100)/50
									# if vc_local_epochs_annealing_factor > 0:
									# 	historical_local_epochs_median = historical_local_epochs_median - vc_local_epochs_annealing_factor
									# 	if historical_local_epochs_median < 4:
									# 		historical_local_epochs_median = 4
									# if community_update_steps_staleness >= historical_staleness_median or \
									# 		vc_local_epochs >= historical_local_epochs_median:
									# 	are_stopping_criteria_reached = True
									if community_update_steps_staleness >= historical_staleness_median:
										are_stopping_criteria_reached = True


								""" THE LOG LOSS CRITERION """
								# validation_loss_log_change = np.log10(current_validation_loss_mvg_avg / previous_validation_loss_mvg_avg)
								# metis_logger.info(msg='%s, Validation Log Loss Change: %s' % (host_id, validation_loss_log_change))
								# if validation_loss_log_change > vc_target_validation_loss_change:
								# 	vc_learner_tombstone += 1
								# 	if vc_learner_tombstone > vc_target_validation_loss_tombstone:
								# 		are_stopping_criteria_reached = True

								""" THE GENERALIZATION LOSS CRITERION """
								# generalization_loss = 100 * ((latest_validation_loss_mean / optimal_validation_loss_mean) - 1)
								# metis_logger.info(msg='Host: %s, GENERALIZATION LOSS: %s' % (host_id, generalization_loss))

								# if generalization_loss >= 0.5:
								# 	are_stopping_criteria_reached = True

								""" THE QUOTIENT OF GENERALIZATION LOSS TO TRAINING PROGRESS CRITERION """
								# training_progress = 1000 * (np.divide(np.sum(train_loss_window), len(train_loss_window) * np.min(train_loss_window)) - 1)
								# generalization_loss_to_progress = np.divide(generalization_loss, training_progress)
								# metis_logger.info(msg='%s, Generalization Loss to Training Progress: %s' % (host_id, generalization_loss_to_progress))

								# if generalization_loss_to_progress >= 0.05:
								# 	are_stopping_criteria_reached = True

						# If the testing variables are defined then we have already feed the network with the testing
						# variables, and now we need to bring it back to its original state, using the trainable variables
						if udf_testing_vars is not None:

							for index, final_var_value in enumerate(final_trainable_vars):
								fed_vars[index].load(final_var_value, mon_sess)
								federated_variables[index].value = final_var_value
							mon_sess.run(assign_atomic_feds)

						# Train learner for a target of burn-in epochs
						if __burnin_epochs < __target_burnin_epochs:
							__burnin_epochs += 1
						else:

							# If learner has a static update frequency then it has surely exited the
							# adaptive execution (Validation Phase)
							if not __learner_in_validation_phase:

								# Since we are in static update frequency mode training data size is the training
								# dataset and the validation set
								host_controller_grpc_client.update_num_training_examples(val=training_dataset_size)
								host_controller_grpc_client.update_num_validation_examples(val=validation_dataset_size)

								""" Static Update Frequency """
								# Check if learner's completed local epochs suffice for a community update request.
								if host_controller_grpc_client.is_eligible_for_async_community_update(
										target_epochs=w_local_epochs):

									learner_model = final_trainable_vars

									if "DVW" in community_function:
										local_model_evaluation_score = host_evaluator_grpc_client\
											.request_model_evaluation(model_variables=learner_model,
																	  is_community_model=False, block=True)
										host_controller_grpc_client.update_latest_validation_score(
											val=local_model_evaluation_score)
									else:
										host_controller_grpc_client.update_latest_validation_score(0.0)
										host_controller_grpc_client.update_num_training_examples(
											val=training_dataset_size + validation_dataset_size)

									host_controller_grpc_client.update_trained_variables(new_variables=learner_model)

									current_update_request_num += 1  # increase number of community update requests
									host_controller_grpc_client.request_current_community(send_learner_state=True,
																						  block=True)
									host_controller_grpc_client.update_target_local_epochs_for_community_update(
										val=w_local_epochs)
									metis_logger.info("Host: {}, Target Update Epochs: {}"
													  .format(host_id, w_local_epochs))

							else:

								""" Adaptive Update Frequency based on Local Validation Loss """
								# Since we are in adaptive execution mode, the training data size is equal only to the
								# training dataset only. Not the validation.
								host_controller_grpc_client.update_num_training_examples(val=training_dataset_size)
								host_controller_grpc_client.update_num_validation_examples(val=validation_dataset_size)

								# If stopping criterion are reached then issue a community update request.
								if __are_stopping_criteria_reached:

									# After much experimentation, it turns out that we must use the latest trained
									# variables instead of the vc_optimal_weights since we incur a performance hit
									# throughout the Validation Phase execution using vc optimal
									# learner_model = vc_optimal_weights
									learner_model = final_trainable_vars

									metis_logger.info(msg="{}, Requesting Federation Validation Score".format(host_id))

									# Capture staleness of current model.
									current_global_epoch_id = \
										host_controller_grpc_client.request_current_global_epoch_id()
									global_update_scalar_clock, learner_global_update_scalar_clock = \
										host_controller_grpc_client.request_community_and_learner_global_scalar_clock()
									community_update_requests_staleness = \
										global_update_scalar_clock - learner_global_update_scalar_clock + 1
									global_community_steps, learner_previous_global_community_steps = \
										host_controller_grpc_client.request_community_and_learner_global_community_steps()
									community_update_steps_staleness = global_community_steps \
																	   - learner_previous_global_community_steps \
																	   + __vc_completed_batches

									if current_global_epoch_id <= global_epoch_transition_threshold:
										local_model_evaluation_score = host_evaluator_grpc_client\
											.request_model_evaluation(model_variables=learner_model,
																	  is_community_model=False, block=True)
										metis_logger.info(msg="{}, Federation Latest Validation Score: {}"
														  .format(host_id, local_model_evaluation_score))
										__vcs_weighting_values[__vc_local_epochs].append(local_model_evaluation_score)

										# The staleness list is associated with the vc_local_epochs list element-by-element
										__vcs_staleness.append(community_update_steps_staleness)
										metis_logger.info(msg="{}. VC Staleness Values: {}"
														  .format(host_id, __vcs_staleness))
										__vcs_local_epochs.append(__vc_local_epochs)
										metis_logger.info(msg="{}. VC Local Epochs: {}"
														  .format(host_id, __vcs_local_epochs))

									else:
										# TODO This is reserved for a transition phase
										pass

									# Reset sentinel variables for the adaptive execution. We also need to reset the
									# training and validation windows on every request
									__vc_train_loss_window = list()
									__vc_validation_loss_window = list()
									__vc_local_epochs = 0
									__vc_completed_batches = 0
									__vc_learner_tombstone = 0
									__vc_optimal_validation_loss = math.inf
									__are_stopping_criteria_reached = False

									# Check whether we have reached the end of the validation phase. If not,
									# then we update community using the FF Score of the local model. If yes, then we
									# need to use the size of the local training data and define a static update
									# frequency (w_target_update_epochs).
									host_controller_grpc_client\
										.update_latest_validation_score(local_model_evaluation_score)
									host_controller_grpc_client\
										.update_trained_variables(new_variables=learner_model)
									host_controller_grpc_client\
										.request_current_community(send_learner_state=True, block=True)
									current_update_request_num += 1

									# Evaluate the community model.
									learner_model = host_controller_grpc_client.retrieve_latest_community_update()
									community_model_evaluation_score = host_evaluator_grpc_client\
										.request_model_evaluation(model_variables=learner_model,
																  is_community_model=True, block=False)


	@classmethod
	def evaluate_federation_model_on_remote_host(cls,
												 fed_host_obj,
												 federated_variables,
												 federation_model_obj,
												 metis_db_session,
												 session_target_stat_name):
		"""
		A function that evaluates the federation model.
		:param fed_host_obj: where to evaluate the model
		:param federated_variables:
		:param federation_model_obj:
		:param metis_db_session:
		:param session_target_stat_name
		:return: None
		"""
		if not any(isinstance(fedvar, fedmodel.FedVar) for fedvar in federated_variables):
			raise TypeError("All the federated variables passed to this function must be of type %s " % fedmodel.FedVar)
		if not isinstance(federation_model_obj, fedmodel.FedModelDef):
			raise TypeError("The 'federation_model_obj' parameter must be of type %s " % fedmodel.FedModelDef)
		if not isinstance(metis_db_session, MetisDBSession):
			raise TypeError("The 'metis_db_session' parameter must be of type %s " % MetisDBSession)

		cluster_spec = fed_host_obj.cluster_spec
		cluster_master = fed_host_obj.fed_master
		worker_server = fed_host_obj.fed_worker_servers[0]
		host_id = fed_host_obj.host_identifier
		w_device_name = worker_server.device_name
		w_is_chief = worker_server.is_leader
		w_config = TFConfiguration.tf_session_config()
		w_device_fn = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=w_device_name)
		w_local_batch_size = worker_server.batch_size

		# Federated Averaging Model Evaluation
		with tf.device(w_device_fn):
			tf.reset_default_graph()
			exec_graph = tf.Graph()
			with exec_graph.as_default():

				# To ensure deterministic operations and random values, helpful for debugging,
				# we need to set the random seed.
				tf.set_random_seed(seed=1990)
				np.random.seed(seed=1990)

				# Import Dataset Structure/Ops.
				_, _, test_dataset_structure = metis_db_session\
					.import_host_data(learner_id=host_id,
									  batch_size=w_local_batch_size,
									  import_train=False,
									  import_validation=False,
									  import_test=True)
				testing_init_op = test_dataset_structure.dataset_init_op
				next_test_dataset = test_dataset_structure.dataset_next
				testing_dataset_size = test_dataset_structure.dataset_size

				# Define Model's input & output format.
				_x_placeholders = federation_model_obj.input_tensors_datatype()
				_y_placeholders = federation_model_obj.output_tensors_datatype()

				# Define Deep NN Graph.
				global_step = tf.train.get_or_create_global_step()
				model_architecture = federation_model_obj.model_architecture(input_tensors=_x_placeholders,
																			 output_tensors=_y_placeholders,
																			 global_step=global_step,
																			 batch_size=w_local_batch_size,
																			 dataset_size=testing_dataset_size)
				# Get predictions from model architecture.
				predictions_tensor_fedmodel = model_architecture.predictions
				# Get trainable variables collection defined in the model graph.
				model_vars = model_architecture.model_federated_variables

				tf_graph_evals = TFGraphEvaluation(_x_placeholders, _y_placeholders, predictions_tensor_fedmodel,
												   model_architecture.loss,
												   is_classification=metis_db_session.is_classification,
												   is_regression=metis_db_session.is_regression,
												   num_classes=metis_db_session.num_classes,
												   negative_classes_indices=metis_db_session.negative_classes_indices,
												   is_eval_output_scalar=metis_db_session.is_eval_output_scalar)
				tf_graph_evals.assign_tfdatasets_operators(None, None, None, None, testing_init_op, next_test_dataset)
				tf_graph_evals.register_evaluation_ops(metis_db_session.get_learner_evaluation_output_attribute(host_id),
													   w_local_batch_size)

				# Create Federated Variables scope.
				with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
					for federated_variable in federated_variables:
						tf.get_variable(name=federated_variable.name,
										initializer=federated_variable.value,
										collections=federated_variable.tf_collection,
										trainable=federated_variable.trainable)

				# Define operations for model weights and biases initialization.
				fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
				assign_atomic_feds = []
				for (index, model_var) in enumerate(model_vars):
					fed_var = fed_vars[index]
					assign_atomic_feds.append(tf.assign(model_var, fed_var))

				master_grpc = cluster_master.grpc_endpoint
				with tf.train.MonitoredTrainingSession(master=master_grpc, is_chief=w_is_chief, config=w_config) as \
						mon_sess:

					# Execute model variables initialization.
					mon_sess.run(assign_atomic_feds)

					_, _, test_eval_results = tf_graph_evals\
							.evaluate_model_on_existing_graph(mon_sess, session_target_stat_name, host_id,
															  include_json_confusion_matrix=True,
															  include_json_evaluation_per_class=True)

		return test_eval_results
