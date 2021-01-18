from experiments.tf_fedmodels.resnet.resnet_cifar_fedmodel import ResNetCifarFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.time_ops import TimeUtil
from collections import OrderedDict

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
CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 100
VALIDATION_PROPORTION = float(os.environ["VALIDATION_PROPORTION"]) if "VALIDATION_PROPORTION" in os.environ else 0.0

# Cluster Setup
# Whether to run on remote servers or multi-localhost
RUN_MULTI_LOCALHOST = True
RUN_MULTI_LOCALHOST_NUM_CLUSTERS = 10

TRAINING_EXAMPLES_NUM = 5000 # ALL: 50000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 1000 # ALL: 10000

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'accuracy'
RUN_WITH_DISTORTED_IMAGES = True

PARTITION_IDS = list(range(10))
W_LOCAL_BATCH_SIZE = 128


def model_execution(tf_fedcluster, learning_rate, momentum):

	data_partitions_num = len(tf_fedcluster.fed_training_hosts)

	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	metis_db_session = MetisDBSession(cifar100_session=True,
									  working_directory="/nfs/isd/stripeli/metis_execution_tfrecords")
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM,
										  dev_examples=DEV_EXAMPLES_NUM,
										  test_examples=TEST_EXAMPLES_NUM,
										  distort_images=RUN_WITH_DISTORTED_IMAGES)
	metis_db_session.partition_session_training_data(partitions_num=len(PARTITION_IDS),
													 classes_per_partition=CLASSES_PER_PARTITION,
													 balanced_class_partitioning=True,
													 skewness_factor=0.0)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nnmodel = ResNetCifarFedModel(num_classes=100,
								  learning_rate=learning_rate,
								  momentum=momentum,
								  resnet_size=50,
								  run_with_distorted_images=RUN_WITH_DISTORTED_IMAGES)
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

			# To ensure deterministic operations and random values, helpful for debugging,
			# we need to set the random seed
			tf.set_random_seed(seed=1990)
			np.random.seed(seed=1990)

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

			partitions_datasets_ops = []
			entire_train_dataset = None
			entire_train_dataset_size = 0
			prefetch_buffer_size = int(W_LOCAL_BATCH_SIZE * 0.1)
			for partition_id in PARTITION_IDS:

				# Import Training Data
				metis_train_dataset, metis_validation_dataset = metis_db_session.session_training_and_holdout_dataset_by_partition_id(partition_id=partition_id,
																											   to_tf_dataset=True,
																											   holdout_proportion=0.0)
				# Create Training Dataset Ops
				partition_train_dataset = metis_train_dataset.data
				partition_validation_dataset = metis_validation_dataset.data
				partition_train_dataset_size = metis_train_dataset.data_size
				entire_train_dataset_size += partition_train_dataset_size

				partition_train_dataset = partition_train_dataset.prefetch(buffer_size=prefetch_buffer_size)
				partition_train_dataset = partition_train_dataset.shuffle(buffer_size=partition_train_dataset_size,
																		  reshuffle_each_iteration=True)
				partition_train_dataset = partition_train_dataset.batch(W_LOCAL_BATCH_SIZE)
				partition_training_iterator = tf.data.Iterator.from_structure(output_types=partition_train_dataset.output_types,
																	output_shapes=partition_train_dataset.output_shapes)
				partition_next_train_dataset = partition_training_iterator.get_next()
				partition_train_dataset_init_op = partition_training_iterator.make_initializer(partition_train_dataset)

				partitions_datasets_ops.append((partition_train_dataset_init_op, partition_next_train_dataset, partition_train_dataset_size))

				partition_validation_iterator = tf.data.Iterator.from_structure(output_types=partition_validation_dataset.output_types,
																				output_shapes=partition_validation_dataset.output_shapes)
				partition_next_validation_dataset = partition_validation_iterator.get_next()
				partition_validation_dataset_init_op = partition_validation_iterator.make_initializer(partition_validation_dataset)

				if entire_train_dataset is None:
					entire_train_dataset = metis_train_dataset.data
				else:
					entire_train_dataset = entire_train_dataset.concatenate(metis_train_dataset.data)

			# Entire Training Dataset
			entire_train_dataset = entire_train_dataset.cache()
			entire_train_dataset = entire_train_dataset.prefetch(buffer_size=prefetch_buffer_size)
			entire_train_dataset = entire_train_dataset.shuffle(buffer_size=entire_train_dataset_size,
																reshuffle_each_iteration=True)
			train_dataset = entire_train_dataset.batch(W_LOCAL_BATCH_SIZE)
			training_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
																output_shapes=train_dataset.output_shapes)
			next_entire_train_dataset = training_iterator.get_next()
			entire_train_dataset_init_op = training_iterator.make_initializer(train_dataset)

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
																				  training_init_op=entire_train_dataset_init_op,
																				  next_train_dataset=next_entire_train_dataset,
																				  validation_init_op=partition_validation_dataset_init_op,
																				  next_validation_dataset=partition_next_validation_dataset,
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

			with tf.train.MonitoredTrainingSession(master=master_grpc, is_chief=True, config=w_config) as mon_sess:

				federation_rounds = list(range(10))
				local_epochs = 4
				for federation_round in federation_rounds:

					clients_models = []
					federation_model_norm_factor = 0
					for partition_idx, partition_dataset_ops in enumerate(partitions_datasets_ops):

						# Initialize trainable variables with the federation values
						for index, original_fed_var in enumerate(federated_variables):
							fed_vars[index].load(original_fed_var.value, mon_sess)
						mon_sess.run(assign_atomic_feds)

						partition_train_dataset_init_op = partition_dataset_ops[0]
						partition_next_train_dataset = partition_dataset_ops[1]
						partition_train_dataset_size = partition_dataset_ops[2]
						federation_model_norm_factor += partition_train_dataset_size

						for local_epoch in range(local_epochs):

							# Re-initialize the iterator when a community update occurs. Start training from scratch.
							mon_sess.run(partition_train_dataset_init_op)

							# Single Epoch Training
							while True:
								try:

									# Load Next Training Batch
									train_batch = mon_sess.run(partition_next_train_dataset)
									train_extra_feeds = OrderedDict()
									for placeholder_name, placeholder_def in _x_placeholders.items():
										train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
									for placeholder_name, placeholder_def in _y_placeholder.items():
										train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

									train_extra_feeds['lr_annealing_value:0'] = 0
									train_extra_feeds['momentum_annealing_value:0'] = 0

									# Train Model
									train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)

								except tf.errors.OutOfRangeError:
									break

						client_model = mon_sess.run(trainable_variables)
						clients_models.append(client_model)
						print("Finished Client Model", partition_idx+1)
						test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																	   x_placeholders=_x_placeholders,
																																	   y_placeholder=_y_placeholder,
																																	   fedhost_eval_ops=fedhost_eval_ops,
																																	   target_stat_name=TARGET_STAT_NAME,
																																	   learner_id=cluster_spec,
																																	   evaluate_validation_set=False)


					print("Finished federation round:", federation_round+1)
					federation_model_matrices = [np.divide(np.multiply(matrix, partitions_datasets_ops[0][2]), federation_model_norm_factor)
												 for matrix in clients_models[0]]
					for client_id, client_model in enumerate(clients_models[1:], start=1):
						for martix_idx, matrix in enumerate(client_model):
							federation_model_matrices[martix_idx] = np.add(
								federation_model_matrices[martix_idx],
								np.divide(
									np.multiply(matrix, partitions_datasets_ops[client_id][2]),
									federation_model_norm_factor
								)
							)
					for idx in range(len(federation_model_matrices)):
						federated_variables[idx].value = federation_model_matrices[idx]

					# Initialize trainable variables with the federation values
					for index, original_fed_var in enumerate(federated_variables):
						fed_vars[index].load(original_fed_var.value, mon_sess)
					mon_sess.run(assign_atomic_feds)

					test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																   x_placeholders=_x_placeholders,
																																   y_placeholder=_y_placeholder,
																																   fedhost_eval_ops=fedhost_eval_ops,
																																   target_stat_name=TARGET_STAT_NAME,
																																   learner_id=cluster_spec,
																																   evaluate_validation_set=False)



	metis_db_session.shutdown()

def init_tf_cluster(clusters_num):
	tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(clusters_num=clusters_num,
																					starting_port=8221)
	return tf_fedcluster


if __name__=="__main__":
	tf_fedcluster = init_tf_cluster(clusters_num=RUN_MULTI_LOCALHOST_NUM_CLUSTERS)
	model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.1, momentum=0.9)