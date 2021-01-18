from experiments.tf_fedmodels.cnn.cnn2_fmnist_model import FmnistFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.time_ops import TimeUtil
from collections import OrderedDict
from collections import defaultdict

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import federation.fed_model as fedmodel
import tensorflow as tf
import numpy as np
import os

np.random.seed(seed=1990)
tf.compat.v1.random.set_random_seed(1990)


scriptDirectory = os.path.dirname(os.path.realpath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["GRPC_VERBOSITY"] = 'ERROR'
CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 10
VALIDATION_PROPORTION = float(os.environ["VALIDATION_PROPORTION"]) if "VALIDATION_PROPORTION" in os.environ else 0.05

# Cluster Setup
# Whether to run on remote servers or multi-localhost
RUN_MULTI_LOCALHOST = True
RUN_MULTI_LOCALHOST_NUM_CLUSTERS = 1

TRAINING_EXAMPLES_NUM = 60000 # ALL: 50000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 10000 # ALL: 10000

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'accuracy'
RUN_WITH_DISTORTED_IMAGES = True

PARTITION_IDS = [0]
W_LOCAL_BATCH_SIZE = 100


def model_execution(tf_fedcluster, learning_rate, momentum):

	data_partitions_num = len(tf_fedcluster.fed_hosts)

	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	metis_db_session = MetisDBSession(fmnist_session=True)
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM,
										  dev_examples=DEV_EXAMPLES_NUM,
										  test_examples=TEST_EXAMPLES_NUM,
										  distort_images=RUN_WITH_DISTORTED_IMAGES)
	metis_db_session.partition_session_training_data(partitions_num=data_partitions_num,
													 classes_per_partition=CLASSES_PER_PARTITION,
													 balanced=True)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nnmodel = FmnistFedModel(learning_rate=learning_rate, momentum=momentum)
	federated_variables = fedmodel.FedModelDef.construct_model_federated_variables(nnmodel)

	host = tf_fedcluster.fed_hosts[0]
	cluster_spec = host.cluster_spec
	cluster_master = host.fed_master
	worker_server = host.fed_worker_servers[0]
	host_training_devices = host.host_training_devices
	w_device_name = worker_server.device_name
	w_config = fedexec.FedExecutionOps.tf_session_config()
	w_device_fn = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=w_device_name)

	with tf.device(w_device_fn):
		tf.reset_default_graph()
		exec_graph = tf.Graph()
		with exec_graph.as_default():

			# Import Testing Data
			metis_test_dataset = metis_db_session.session_testing_dataset(to_tf_dataset=True)

			# Create Testing Dataset Ops
			test_dataset = metis_test_dataset
			test_dataset = test_dataset.batch(W_LOCAL_BATCH_SIZE)
			testing_iterator = tf.data.Iterator.from_structure(output_types=test_dataset.output_types,
															   output_shapes=test_dataset.output_shapes)
			next_test_dataset = testing_iterator.get_next()
			test_dataset_init_op = testing_iterator.make_initializer(test_dataset)

			# If we want to evaluate model on a large number of partitions to resemble convergence in a federation setting.
			# For instance, in the centralized case we might need to evaluate using 50% of the data, we can create a dataset out of 5 partitions.

			train_dataset = None
			validation_dataset = None
			train_dataset_size = 0
			validation_dataset_size = 0
			for partition_id in PARTITION_IDS:

				# Import Training Data
				metis_train_dataset, metis_validation_dataset = metis_db_session.session_training_and_holdout_dataset_by_partition_id(partition_id=partition_id,
																																	  to_tf_dataset=True,
																																	  holdout_proportion=0.0)
				# Create Training Dataset Ops
				partition_train_dataset = metis_train_dataset.data
				partition_validation_dataset = metis_validation_dataset.data
				if train_dataset is None and validation_dataset is None:
					train_dataset = partition_train_dataset
					validation_dataset = partition_validation_dataset
				else:
					train_dataset = train_dataset.concatenate(partition_train_dataset)
					validation_dataset = validation_dataset.concatenate(partition_validation_dataset)

				partition_train_dataset_size = metis_train_dataset.data_size
				train_dataset_size += partition_train_dataset_size
				partition_validation_dataset_size = metis_validation_dataset.data_size
				validation_dataset_size += partition_validation_dataset_size

			prefetch_buffer_size = int(W_LOCAL_BATCH_SIZE * 0.1)
			train_dataset = train_dataset.cache()
			train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)
			train_dataset = train_dataset.shuffle(buffer_size=train_dataset_size, reshuffle_each_iteration=True)
			train_dataset = train_dataset.batch(W_LOCAL_BATCH_SIZE)
			training_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
																output_shapes=train_dataset.output_shapes)
			next_train_dataset = training_iterator.get_next()
			train_dataset_init_op = training_iterator.make_initializer(train_dataset)

			# Create Validation Dataset Ops
			validation_iterator = tf.data.Iterator.from_structure(output_types=validation_dataset.output_types,
																  output_shapes=validation_dataset.output_shapes)
			validation_dataset_init_op = validation_iterator.make_initializer(validation_dataset)
			next_validation_dataset = validation_iterator.get_next()

			# Define Model's input & output format
			_x_placeholders = nnmodel.input_tensors_datatype()
			_y_placeholder = nnmodel.output_tensors_datatype()

			# Define model's global step, size of the host's local batch and dataset
			_global_step = tf.train.get_or_create_global_step()

			# Define Deep NN Graph
			model_architecture = nnmodel.model_architecture(input_tensors=_x_placeholders,
															output_tensors=_y_placeholder,
															global_step=_global_step,
															batch_size=W_LOCAL_BATCH_SIZE,
															dataset_size=train_dataset_size,
															learner_training_devices=host_training_devices)

			train_step = model_architecture.train_step
			loss_tensor_fedmodel = model_architecture.loss
			predictions_tensor_fedmodel = model_architecture.predictions

			fedhost_eval_ops = fedexec.FedExecutionOps.register_evaluation_tf_ops(metis_db_session=metis_db_session,
																				  x_placeholders=_x_placeholders,
																				  y_placeholder=_y_placeholder,
																				  loss_tensor_fedmodel=loss_tensor_fedmodel,
																				  predictions_tensor_fedmodel=predictions_tensor_fedmodel,
																				  batch_size=W_LOCAL_BATCH_SIZE,
																				  training_init_op=train_dataset_init_op,
																				  next_train_dataset=next_train_dataset,
																				  validation_init_op=validation_dataset_init_op,
																				  next_validation_dataset=next_validation_dataset,
																				  testing_init_op=test_dataset_init_op,
																				  next_test_dataset=next_test_dataset)

			# Get trainable variables collection defined in the model graph
			trainable_variables = model_architecture.model_federated_variables

			with tf.variable_scope("fedvars_atomic", reuse=tf.AUTO_REUSE):
				for variable in federated_variables:
					tf.get_variable(name=variable.name, initializer=variable.value, collections=variable.tf_collection, trainable=variable.trainable)

			# Define operations for model weights and biases initialization
			fed_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fedvars_atomic')
			assign_atomic_feds = []
			for (index, model_var) in enumerate(trainable_variables):
				fed_var = fed_vars[index]
				assign_atomic_feds.append(tf.assign(model_var, fed_var))

			# is_chief controls whether we want a Master or a Worker Training Session
			master_grpc = cluster_master.grpc_endpoint

			# Experiment Configs
			epoch_test_evaluations = OrderedDict()
			test_evaluations = defaultdict(list)
			execution_times = defaultdict(list)
			train_data_pct = [1] # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
			local_epochs = [100] #, 15, 20, 25, 30, 35, 40, 45, 50] #, 75, 100, 200, 500]
			num_runs = 1

			with tf.train.MonitoredTrainingSession(master=master_grpc, is_chief=True, config=w_config) as mon_sess:

				for run_data_pct in train_data_pct:
					for run_id in range(0, num_runs):
						for run_local_epochs in local_epochs:

							total_batches = train_dataset_size / W_LOCAL_BATCH_SIZE

							# Initialize trainable variables with the federation values
							for index, original_fed_var in enumerate(federated_variables):
								fed_vars[index].load(original_fed_var.value, mon_sess)
							mon_sess.run(assign_atomic_feds)
							previous_epoch_variables_flatten = np.hstack([var.flatten() for var in mon_sess.run(trainable_variables)])

							metis_logger.info(msg="Run ID: {}, Data Pct: {}, Local Epochs: {}".format(run_id+1, run_data_pct, run_local_epochs))
							metis_logger.info(msg="Before Training")
							test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																		   x_placeholders=_x_placeholders,
																																		   y_placeholder=_y_placeholder,
																																		   fedhost_eval_ops=fedhost_eval_ops,
																																		   target_stat_name=TARGET_STAT_NAME,
																																		   learner_id=cluster_spec,
																																		   evaluate_validation_set=False)
							for epoch_id in range(0, run_local_epochs):

								# if (epoch_id+1) % bagging_models_window == 0:
								# 	# Initialize trainable variables with the federation values
								# 	mon_sess.run(assign_atomic_feds)

								# Re-initialize the iterator when a community update occurs. Start training from scratch.
								mon_sess.run(train_dataset_init_op)

								epoch_batch_id = 0
								e_start_time = TimeUtil.current_milli_time()

								# Single Epoch Training
								while True:
									try:

										# Load Next Training Batch
										train_batch = mon_sess.run(next_train_dataset)
										train_extra_feeds = OrderedDict()
										for placeholder_name, placeholder_def in _x_placeholders.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
										for placeholder_name, placeholder_def in _y_placeholder.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

										# Train Model
										train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)

										epoch_batch_id += 1

									except tf.errors.OutOfRangeError:
										break

								print("Epoch: {}".format(epoch_id + 1))
								test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																			   x_placeholders=_x_placeholders,
																																			   y_placeholder=_y_placeholder,
																																			   fedhost_eval_ops=fedhost_eval_ops,
																																			   target_stat_name=TARGET_STAT_NAME,
																																			   learner_id=cluster_spec,
																																			   evaluate_validation_set=False)
								epoch_test_evaluations[epoch_id + 1] = test_eval_results['accuracy']
								e_end_time = TimeUtil.current_milli_time()
								e_duration = TimeUtil.delta_diff_in_secs(e_start_time, e_end_time)
								execution_times[(run_local_epochs, run_data_pct)].append(e_duration)
								metis_logger.info(msg="Epoch Duration: {} secs for {}% of data".format(e_duration, run_data_pct*100))


								if epoch_id % 5 == 0:
									current_epoch_variables_flatten = np.hstack([var.flatten() for var in mon_sess.run(trainable_variables)])

									models_dot_product = np.dot(previous_epoch_variables_flatten, current_epoch_variables_flatten)
									previous_model_norm = np.sqrt(np.sum(np.power(previous_epoch_variables_flatten, 2)))
									current_model_norm = np.sqrt(np.sum(np.power(current_epoch_variables_flatten, 2)))
									norm = np.multiply(previous_model_norm, current_model_norm)
									cosine_sim = np.divide(models_dot_product, norm)
									print("Cosine Similarity: ", cosine_sim)
									previous_epoch_variables_flatten = current_epoch_variables_flatten




							metis_logger.info(msg="Final Model:")
							test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																		   x_placeholders=_x_placeholders,
																																		   y_placeholder=_y_placeholder,
																																		   fedhost_eval_ops=fedhost_eval_ops,
																																		   target_stat_name=TARGET_STAT_NAME,
																																		   learner_id=cluster_spec,
																																		   evaluate_validation_set=False)

def init_tf_cluster(clusters_num):
	tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(hostname="bdnf.isi.edu",
																					clusters_num=clusters_num,
																					starting_port=9221)
	return tf_fedcluster


if __name__=="__main__":
	tf_fedcluster = init_tf_cluster(clusters_num=RUN_MULTI_LOCALHOST_NUM_CLUSTERS)
	model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.01, momentum=0.9)