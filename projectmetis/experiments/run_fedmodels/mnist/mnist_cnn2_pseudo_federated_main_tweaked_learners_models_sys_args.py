from experiments.tf_fedmodels.cnn.cnn2_mnist_model import MnistFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.time_ops import TimeUtil
from collections import OrderedDict

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import federation.fed_model as fedmodel
import tensorflow as tf
import numpy as np
import random
import os

random.seed(1990)
np.random.seed(seed=1990)
tf.set_random_seed(seed=1990)


scriptDirectory = os.path.dirname(os.path.realpath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["GRPC_VERBOSITY"] = 'ERROR'
CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 1
VALIDATION_PROPORTION = float(os.environ["VALIDATION_PROPORTION"]) if "VALIDATION_PROPORTION" in os.environ else 0.0

# Cluster Setup
# Whether to run on remote servers or multi-localhost
RUN_MULTI_LOCALHOST = True
RUN_MULTI_LOCALHOST_NUM_CLUSTERS = 1

TRAINING_EXAMPLES_NUM = 60000 #12665 # ALL: 60000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 10000 # ALL: 10000

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'accuracy'
RUN_WITH_DISTORTED_IMAGES = True

PARTITION_IDS = list(range(10))
W_LOCAL_BATCH_SIZE = 100


FOUT_NAME = "mnist.classes1.x10Learners.x200FederationRounds.withFCLayerTweak.csv"


def model_execution(tf_fedcluster, learning_rate):

	data_partitions_num = len(PARTITION_IDS)

	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	metis_db_session = MetisDBSession(mnist_session=True, working_directory="/nfs/isd/stripeli/metis_execution_tfrecords/")
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM,
										  dev_examples=DEV_EXAMPLES_NUM,
										  test_examples=TEST_EXAMPLES_NUM)
	metis_db_session.partition_session_training_data(partitions_num=data_partitions_num,
													 balanced_class_partitioning=True,
													 classes_per_partition=CLASSES_PER_PARTITION)
	partitions_classes_stats = metis_db_session.retrieve_all_partitions_datasets_stats(to_json_representation=True)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nnmodel = MnistFedModel(learning_rate=learning_rate)
	federated_variables = fedmodel.FedModelDef.construct_model_federated_variables(nnmodel)

	# npzfile = np.load(scriptDirectory + "/../run_fedmodels/cifar/cnn2_cached_models/IFVL_async_federation_cnn2_model_0.7947.npz") # Test Accuracy: 0.7947
	# npzfile = np.load(scriptDirectory + "/../run_fedmodels/cifar/cnn2_cached_models/IFVL_async_federation_cnn2_model_0.8089.npz") # Test Accuracy: 0.8089
	# npzfile = np.load(scriptDirectory + "/../run_fedmodels/cifar/cnn2_cached_models/IFVL_async_federation_cnn2_model_0.8139.npz") # Test Accuracy: 0.8139
	# arrays_ids = list(npzfile.keys())
	# for idx in range(len(federated_variables)):
	# 	array_value = npzfile[arrays_ids[idx]]
	# 	federated_variables[idx].value = array_value

	host = tf_fedcluster.fed_training_hosts[0]
	cluster_spec = host.cluster_spec
	cluster_master = host.fed_master
	worker_server = host.fed_worker_servers[0]
	host_training_devices = host.host_training_devices
	w_device_name = worker_server.device_name
	w_config = fedexec.FedExecutionOps.tf_session_config()
	w_device_fn = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=w_device_name)

	# Experiment Configs
	all_classes = set(range(metis_db_session.num_classes))
	federation_rounds = 50
	local_epochs = 4
	community_norm_factor = len(PARTITION_IDS)
	PARTITION_IDS.append("community_model")
	partition_model_weight = 1
	community_model = [var.value for var in federated_variables]
	community_model_fedrounds_evaluation = dict()
	community_model_mask_ids = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

	for federation_round_id in range(federation_rounds):

		federation_round_partitions_models = []
		for partition_id in PARTITION_IDS:
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

					"""
					If we want to evaluate model on a large number of partitions to resemble convergence in a federation setting.
					For instance, in the centralized case we might need to evaluate using 50% of the data, we can create a dataset out of 5 partitions.
					"""
					# Import Training Data
					assinging_partition_id = partition_id
					if "community_model" == partition_id:
						assinging_partition_id = 0
					metis_train_dataset, metis_validation_dataset = metis_db_session.session_training_and_holdout_dataset_by_partition_id(partition_id=assinging_partition_id,
																																		  to_tf_dataset=True,
																																		  holdout_proportion=0.0)
					# Create Training Dataset Ops
					train_dataset = metis_train_dataset.data
					validation_dataset = metis_validation_dataset.data

					partition_train_dataset_size = metis_train_dataset.data_size
					train_dataset_size = partition_train_dataset_size
					partition_validation_dataset_size = metis_validation_dataset.data_size
					validation_dataset_size = partition_validation_dataset_size

					prefetch_buffer_size = int(W_LOCAL_BATCH_SIZE * 0.1)
					train_dataset = train_dataset.cache()
					train_dataset = train_dataset.prefetch(buffer_size=prefetch_buffer_size)
					train_dataset = train_dataset.shuffle(buffer_size=train_dataset_size, reshuffle_each_iteration=True)
					train_dataset = train_dataset.batch(W_LOCAL_BATCH_SIZE)
					training_iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
																		output_shapes=train_dataset.output_shapes)
					next_train_dataset_batch = training_iterator.get_next()
					train_dataset_init_op = training_iterator.make_initializer(train_dataset)

					# Create Validation Dataset Ops
					validation_iterator = tf.data.Iterator.from_structure(output_types=validation_dataset.output_types,
																		  output_shapes=validation_dataset.output_shapes)
					validation_dataset_init_op = validation_iterator.make_initializer(validation_dataset)
					next_validation_dataset_batch = validation_iterator.get_next()

					# Define Model's input & output format
					_x_placeholders = nnmodel.input_tensors_datatype()
					_y_placeholder = nnmodel.output_tensors_datatype()

					# Define model's global step, size of the host's local batch and dataset
					_global_step = tf.train.get_or_create_global_step()

					class_ids_mask = community_model_mask_ids
					# if "community_model" == partition_id:
					# 	class_ids_mask = community_model_mask_ids
					# else:
					# 	# Zeroing last layer nodes according to partition class ids
					# 	partition_classes = set(partitions_classes_stats[partition_id]['partition_classes'])
					# 	class_ids_mask = [0.0] * len(all_classes)
					# 	for cid in partition_classes:
					# 		class_ids_mask[cid] = 1.0

					# Define Deep NN Graph
					model_architecture = nnmodel.model_architecture(input_tensors=_x_placeholders,
																	output_tensors=_y_placeholder,
																	class_ids_mask=class_ids_mask,
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
																						  next_train_dataset=next_train_dataset_batch,
																						  validation_init_op=validation_dataset_init_op,
																						  next_validation_dataset=next_validation_dataset_batch,
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

						if not "community_model" == partition_id:
							# Initialize each partition trainable variables with the federation variables
							for index, original_fed_var in enumerate(federated_variables):
								assigning_value = original_fed_var.value
								fed_vars[index].load(assigning_value, mon_sess)
							mon_sess.run(assign_atomic_feds)

							# class_ids_mask = tf.get_default_graph().get_tensor_by_name("fc2/class_ids_mask:0")
							# print(class_ids_mask, mon_sess.run(class_ids_mask))
							# y_conv_with_cid_mask = tf.get_default_graph().get_tensor_by_name("fc2/y_conv_with_cid_mask:0")
							# print(y_conv_with_cid_mask)

							for epoch_id in range(0, local_epochs):

								# Re-initialize the iterator when a community update occurs. Start training from scratch.
								mon_sess.run(train_dataset_init_op)

								epoch_batch_id = 0

								# Single Epoch Training
								while True:
									try:

										# Load Next Training Batch
										train_batch = mon_sess.run(next_train_dataset_batch)
										train_extra_feeds = OrderedDict()
										for placeholder_name, placeholder_def in _x_placeholders.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
										for placeholder_name, placeholder_def in _y_placeholder.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

										train_extra_feeds['lr_value:0'] = learning_rate
										train_extra_feeds['momentum_value:0'] = 0
										# train_extra_feeds['lr_annealing_value:0'] = 0
										# train_extra_feeds['momentum_annealing_value:0'] = 0

										# Train Model
										train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)

										epoch_batch_id += 1

									except tf.errors.OutOfRangeError:
										break

							# print("Y_CONV with cid mask after last training batch:", mon_sess.run(y_conv_with_cid_mask, feed_dict=train_extra_feeds))
							partition_model = mon_sess.run(trainable_variables)
							federation_round_partitions_models.append(partition_model)
							print("\nPartition Model: {} Evaluation After {} Epochs".format(partition_id+1, local_epochs))
							test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																		   x_placeholders=_x_placeholders,
																																		   y_placeholder=_y_placeholder,
																																		   fedhost_eval_ops=fedhost_eval_ops,
																																		   target_stat_name=TARGET_STAT_NAME,
																																		   learner_id=cluster_spec,
																																		   evaluate_validation_set=False)

						else:
							previous_community_model = community_model
							community_model = [partition_model_weight/community_norm_factor * matrix
												for matrix in federation_round_partitions_models[0]]
							for model in federation_round_partitions_models[1:]:
								for m_idx, matrix in enumerate(model):
									new_matrix = partition_model_weight/community_norm_factor * matrix
									community_model[m_idx] += new_matrix
							for index, final_var_value in enumerate(community_model):
								fed_vars[index].load(final_var_value, mon_sess)
								federated_variables[index].value = final_var_value
							mon_sess.run(assign_atomic_feds)
							print("\nFederation Round {}, Community Model Evaluation:".format(federation_round_id+1))
							test_eval_results, train_eval_results, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																		   x_placeholders=_x_placeholders,
																																		   y_placeholder=_y_placeholder,
																																		   fedhost_eval_ops=fedhost_eval_ops,
																																		   target_stat_name=TARGET_STAT_NAME,
																																		   learner_id=cluster_spec,
																																		   evaluate_validation_set=False)
							community_model_fedrounds_evaluation[federation_round_id+1] = test_eval_results['accuracy']


	with open(FOUT_NAME, 'w+') as fout:
		fline = "federation_round_id, test_accuracy\n"
		fout.write(fline)
		for fid in sorted(community_model_fedrounds_evaluation.keys()):
			fout.write("{}, {}\n".format(fid, community_model_fedrounds_evaluation[fid]))


def init_tf_cluster(clusters_num):
	tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(clusters_num=clusters_num,
																					starting_port=10221)
	return tf_fedcluster


if __name__=="__main__":
	tf_fedcluster = init_tf_cluster(clusters_num=RUN_MULTI_LOCALHOST_NUM_CLUSTERS)
	# model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.0015)
	model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.01)