from experiments.tf_fedmodels.cnn.cnn2_mnist_model import MnistFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.time_ops import TimeUtil
from collections import OrderedDict

from hyperopt import hp, fmin, tpe, Trials, space_eval
import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import federation.fed_model as fedmodel
import hyperopt as hppt
import tensorflow as tf
import pickle
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

TRAINING_EXAMPLES_NUM = 60000 # ALL: 60000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 10000 # ALL: 10000

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'accuracy'
RUN_WITH_DISTORTED_IMAGES = True

DATA_PARTITION_IDS = list(range(10))
W_LOCAL_BATCH_SIZE = 100

# Experiment Configs
FEDERATION_ROUNDS = 50
LOCAL_EPOCHS = 4


FOUT_NAME = "mnist.classes{}.x10Learners.PseudoFederated.FedRounds{}.csv".format(
	CLASSES_PER_PARTITION, FEDERATION_ROUNDS)


def model_execution(tf_fedcluster, learning_rate, momentum):

	data_partitions_num = len(DATA_PARTITION_IDS)

	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	# metis_db_session = MetisDBSession(cifar10_session=True, working_directory="/nfs/isd/stripeli/metis_execution_tfrecords/")
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
	
	RUNNING_PARTITIONS_IDS = DATA_PARTITION_IDS.copy() # duplicate list
	RUNNING_PARTITIONS_IDS.append("COMMUNITY_MODEL_COMPUTATION")
	community_model_fedrounds_evaluation = dict()


	for federation_round_id in range(FEDERATION_ROUNDS):
		
		federation_round_partitions_models = []

		# Loop for each partition perform local training
		for running_partition_id in RUNNING_PARTITIONS_IDS:

			with tf.device(w_device_fn):
				tf.reset_default_graph()
				exec_graph = tf.Graph()
				with exec_graph.as_default():

					# Define Model's input & output format
					_x_placeholders = nnmodel.input_tensors_datatype()
					_y_placeholder = nnmodel.output_tensors_datatype()

					# Define model's global step, size of the host's local batch and dataset
					_global_step = tf.train.get_or_create_global_step()

					# Define Deep NN Graph
					model_architecture = nnmodel.model_architecture(input_tensors=_x_placeholders,
																	output_tensors=_y_placeholder)
					train_step = model_architecture.train_step
					loss_tensor_fedmodel = model_architecture.loss
					predictions_tensor_fedmodel = model_architecture.predictions

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

					# Import Testing Data
					metis_test_dataset = metis_db_session.session_testing_dataset(to_tf_dataset=True)

					# Create Testing Dataset Ops
					test_dataset = metis_test_dataset
					test_dataset = test_dataset.batch(W_LOCAL_BATCH_SIZE)
					testing_iterator = tf.data.Iterator.from_structure(output_types=test_dataset.output_types,
																	   output_shapes=test_dataset.output_shapes)
					next_test_dataset = testing_iterator.get_next()
					test_dataset_init_op = testing_iterator.make_initializer(test_dataset)


					if running_partition_id != "COMMUNITY_MODEL_COMPUTATION":

						"""
						If we want to evaluate model on a large number of partitions to resemble convergence in a federation setting.
						For instance, in the centralized case we might need to evaluate using 50% of the data, we can create a dataset out of 5 partitions.
						"""
						partitions_total_train_dataset_size = 0
						partitions_total_validation_dataset_size = 0
						partitions_distributed_training_dataset = None
						partitions_distributed_validation_dataset = None

						# Generate current partition's local datasets and distributed datasets
						for data_partition_id in DATA_PARTITION_IDS:

							# Import Training Data
							metis_train_dataset, metis_validation_dataset = metis_db_session.session_training_and_holdout_dataset_by_partition_id(partition_id=data_partition_id,
																																				  to_tf_dataset=True,
																																				  holdout_proportion=VALIDATION_PROPORTION)
							partition_train_dataset = metis_train_dataset.data
							partition_validation_dataset = metis_validation_dataset.data

							partition_train_dataset_size = metis_train_dataset.data_size
							partitions_total_train_dataset_size += partition_train_dataset_size

							partition_validation_dataset_size = metis_validation_dataset.data_size
							partitions_total_validation_dataset_size += partition_validation_dataset_size

							# Concatenate Validation Datasets to create the Distributed Validation Dataset
							if partitions_distributed_training_dataset is None and partitions_distributed_validation_dataset is None:
								partitions_distributed_training_dataset = partition_train_dataset
								partitions_distributed_validation_dataset = partition_validation_dataset
							else:
								partitions_distributed_training_dataset.concatenate(partition_train_dataset)
								partitions_distributed_validation_dataset.concatenate(partition_validation_dataset)

							if running_partition_id == data_partition_id:

								# Create Training Dataset Ops
								# partition_model_federation_weight = 1
								partition_model_federation_weight = partition_train_dataset_size

								prefetch_buffer_size = int(W_LOCAL_BATCH_SIZE * 0.1)
								partition_train_dataset = partition_train_dataset.cache()
								partition_train_dataset = partition_train_dataset.prefetch(buffer_size=prefetch_buffer_size)
								partition_train_dataset = partition_train_dataset.shuffle(buffer_size=partitions_total_train_dataset_size,
																						  reshuffle_each_iteration=True)
								partition_train_dataset = partition_train_dataset.batch(W_LOCAL_BATCH_SIZE)
								partition_training_iterator = tf.data.Iterator.from_structure(output_types=partition_train_dataset.output_types,
																							  output_shapes=partition_train_dataset.output_shapes)
								partition_train_dataset_init_op = partition_training_iterator.make_initializer(partition_train_dataset)
								partition_train_dataset_next_batch_op = partition_training_iterator.get_next()

								# Create Validation Dataset Ops
								partition_validation_dataset = partition_validation_dataset.batch(W_LOCAL_BATCH_SIZE)
								partition_validation_iterator = tf.data.Iterator.from_structure(output_types=partition_validation_dataset.output_types,
																								output_shapes=partition_validation_dataset.output_shapes)
								partition_validation_dataset_init_op = partition_validation_iterator.make_initializer(partition_validation_dataset)
								partition_validation_dataset_next_batch_op = partition_validation_iterator.get_next()

								fedhost_eval_ops = fedexec.FedExecutionOps.register_evaluation_tf_ops(metis_db_session=metis_db_session,
																									  x_placeholders=_x_placeholders,
																									  y_placeholder=_y_placeholder,
																									  loss_tensor_fedmodel=loss_tensor_fedmodel,
																									  predictions_tensor_fedmodel=predictions_tensor_fedmodel,
																									  batch_size=W_LOCAL_BATCH_SIZE,
																									  training_init_op=partition_train_dataset_init_op,
																									  next_train_dataset=partition_train_dataset_next_batch_op,
																									  validation_init_op=partition_validation_dataset_init_op,
																									  next_validation_dataset=partition_validation_dataset_next_batch_op,
																									  testing_init_op=test_dataset_init_op,
																									  next_test_dataset=next_test_dataset)

						partitions_distributed_validation_dataset = partitions_distributed_validation_dataset.batch(W_LOCAL_BATCH_SIZE)
						partitions_distributed_validation_iterator = tf.data.Iterator.from_structure(output_types=partitions_distributed_validation_dataset.output_types,
																									 output_shapes=partitions_distributed_validation_dataset.output_shapes)
						partitions_distributed_validation_dataset_init_op = partitions_distributed_validation_iterator.make_initializer(partitions_distributed_validation_dataset)
						partitions_distributed_validation_dataset_next_batch_op = partitions_distributed_validation_iterator.get_next()
						partitions_distributed_validation_hpopt_eval = fedexec.FedExecutionOps.register_evaluation_tf_ops(metis_db_session=metis_db_session,
																														  x_placeholders=_x_placeholders,
																														  y_placeholder=_y_placeholder,
																														  loss_tensor_fedmodel=loss_tensor_fedmodel,
																														  predictions_tensor_fedmodel=predictions_tensor_fedmodel,
																														  batch_size=W_LOCAL_BATCH_SIZE,
																														  training_init_op=partition_train_dataset_init_op,
																														  next_train_dataset=partition_train_dataset_next_batch_op,
																														  validation_init_op=partitions_distributed_validation_dataset_init_op,
																														  next_validation_dataset=partitions_distributed_validation_dataset_next_batch_op,
																														  testing_init_op=test_dataset_init_op,
																														  next_test_dataset=next_test_dataset)

						# community_norm_factor = len(PARTITION_IDS)
						community_norm_factor = partitions_total_train_dataset_size

						with tf.train.MonitoredTrainingSession(master=master_grpc, is_chief=True, config=w_config) as mon_sess:

							# Initialize each partition trainable variables with the federation variables
							for index, fed_var in enumerate(federated_variables):
								assigning_value = fed_var.value
								fed_vars[index].load(assigning_value, mon_sess)
							mon_sess.run(assign_atomic_feds)

							def single_epoch_training(learning_rate, momentum=0.0, perform_epoch_evaluation=False):

								# Re-initialize the iterator when a community update occurs. Start training from scratch.
								mon_sess.run(partition_train_dataset_init_op)

								# Single Epoch Training
								while True:
									try:

										# Load Next Training Batch
										train_batch = mon_sess.run(partition_train_dataset_next_batch_op)
										train_extra_feeds = OrderedDict()
										for placeholder_name, placeholder_def in _x_placeholders.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
										for placeholder_name, placeholder_def in _y_placeholder.items():
											train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

										train_extra_feeds['lr_value:0'] = learning_rate
										train_extra_feeds['momentum_value:0'] = momentum
										# train_extra_feeds['lr_annealing_value:0'] = 0
										# train_extra_feeds['momentum_annealing_value:0'] = 0

										# Train Model
										train_step.run_tf_operation(session=mon_sess, extra_feeds=train_extra_feeds)

									except tf.errors.OutOfRangeError:
										break

								if perform_epoch_evaluation:

									# Evaluate Against Distributed Validation Dataset
									fedhost_eval_ops = partitions_distributed_validation_hpopt_eval

									_, _, validation_eval_results = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																																		x_placeholders=_x_placeholders,
																																		y_placeholder=_y_placeholder,
																																		fedhost_eval_ops=fedhost_eval_ops,
																																		target_stat_name=TARGET_STAT_NAME,
																																		learner_id=cluster_spec,
																																		evaluate_validation_set=True)
									mon_sess.run(assign_atomic_feds)
									return validation_eval_results

								else:
									print(learning_rate)



							def hyper_parameters_optimization():

								def objective_fn(hyperparameters):

									fed_lr_value = float(hyperparameters['learning_rate'])
									eval_results = single_epoch_training(learning_rate=fed_lr_value, perform_epoch_evaluation=True)
									validation_loss_mean = eval_results['validation_loss_mean']
									print("LR VAL: {}, Validation Loss Mean: {}".format(fed_lr_value, validation_loss_mean))
									results = {
										'loss': validation_loss_mean,
										'status': hppt.STATUS_OK
									}
									return results

								hyperparams_space = {
									'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.02)),
								}

								max_trials = 5
								trials_step = 1  # how many additional trials to do after loading saved trials.

								hp_trials_fout = "learner{}_trials.hyperopt".format(running_partition_id)
								try:  # try to load an already saved trials object, and increase the max
									bayes_trials = pickle.load(open(hp_trials_fout, "rb"))
									print("Found saved Trials! Loading...")
									max_trials = len(bayes_trials.trials) + trials_step
									print("Rerunning from {} trials to {} (+{}) trials".format(len(bayes_trials.trials), max_trials, trials_step))
								except:  # create a new trials object and start searching
									bayes_trials = Trials()

								# Optimize
								best_config = fmin(fn=objective_fn,
												   space=hyperparams_space,
												   algo=tpe.suggest,
												   max_evals=max_trials,
												   trials=bayes_trials)
								best_config_eval = space_eval(hyperparams_space, best_config)
								print("BEST Configuration for Learner {}: {}".format(running_partition_id+1, best_config_eval))

								# save the trials object
								with open(hp_trials_fout, "wb") as f:
									pickle.dump(bayes_trials, f)

								return best_config_eval

							# print("##########################################")
							# print("HP OPT for learner:", running_partition_id+1)
							# best_hyper_param_config = hyper_parameters_optimization()
							# best_learning_rate = best_hyper_param_config['learning_rate']
							# print("##########################################")

							# mon_sess.run(assign_atomic_feds)
							for epoch_id in range(0, LOCAL_EPOCHS):

								# single_epoch_training(learning_rate=best_learning_rate, momentum=momentum, perform_epoch_evaluation=False)
								single_epoch_training(learning_rate=learning_rate, momentum=momentum, perform_epoch_evaluation=False)

							print("\nPartition Model: {} Evaluation After {} Epochs".format(running_partition_id+1, LOCAL_EPOCHS))
							test_eval_results, _, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																														  x_placeholders=_x_placeholders,
																														  y_placeholder=_y_placeholder,
																														  fedhost_eval_ops=fedhost_eval_ops,
																														  target_stat_name=TARGET_STAT_NAME,
																														  learner_id=cluster_spec,
																														  evaluate_validation_set=True)
							print("\n")

							partition_model = mon_sess.run(trainable_variables)
							weighted_partition_model = []
							weighting_factor = partition_model_federation_weight/community_norm_factor
							for matrix in partition_model:
								weighted_partition_model.append(weighting_factor*matrix)
							federation_round_partitions_models.append(weighted_partition_model)

					else:

						# Import Training Data
						metis_train_dataset, metis_validation_dataset = metis_db_session.session_training_and_holdout_dataset_by_partition_id(partition_id=0,
																																			  to_tf_dataset=True,
																																			  holdout_proportion=VALIDATION_PROPORTION)
						partition_train_dataset = metis_train_dataset.data
						partition_validation_dataset = metis_validation_dataset.data

						partition_train_dataset = partition_train_dataset.batch(W_LOCAL_BATCH_SIZE)
						partition_training_iterator = tf.data.Iterator.from_structure(output_types=partition_train_dataset.output_types,
																					  output_shapes=partition_train_dataset.output_shapes)
						partition_train_dataset_init_op = partition_training_iterator.make_initializer(partition_train_dataset)
						partition_train_dataset_next_batch_op = partition_training_iterator.get_next()

						partition_validation_dataset = partition_validation_dataset.batch(W_LOCAL_BATCH_SIZE)
						partition_validation_iterator = tf.data.Iterator.from_structure(output_types=partition_validation_dataset.output_types,
																					  output_shapes=partition_validation_dataset.output_shapes)
						partition_validation_dataset_init_op = partition_validation_iterator.make_initializer(partition_train_dataset)
						partition_validation_dataset_next_batch_op = partition_validation_iterator.get_next()

						fedhost_eval_ops = fedexec.FedExecutionOps.register_evaluation_tf_ops(metis_db_session=metis_db_session,
																							  x_placeholders=_x_placeholders,
																							  y_placeholder=_y_placeholder,
																							  loss_tensor_fedmodel=loss_tensor_fedmodel,
																							  predictions_tensor_fedmodel=predictions_tensor_fedmodel,
																							  batch_size=W_LOCAL_BATCH_SIZE,
																							  training_init_op=partition_train_dataset_init_op,
																							  next_train_dataset=partition_train_dataset_next_batch_op,
																							  validation_init_op=partition_validation_dataset_init_op,
																							  next_validation_dataset=partition_validation_dataset_next_batch_op,
																							  testing_init_op=test_dataset_init_op,
																							  next_test_dataset=next_test_dataset)

						with tf.train.MonitoredTrainingSession(master=master_grpc, is_chief=True, config=w_config) as mon_sess:


							community_model = federation_round_partitions_models[0]
							for weighted_partition_model in federation_round_partitions_models[1:]:
								for m_idx, matrix in enumerate(weighted_partition_model):
									community_model[m_idx] += matrix
							for index, community_matrix in enumerate(community_model):
								fed_vars[index].load(community_matrix, mon_sess)
								federated_variables[index].value = community_matrix
							mon_sess.run(assign_atomic_feds)
							metis_logger.info(msg="\nFederation Round {}, Community Model Evaluation:".format(federation_round_id+1))
							test_eval_results, _, _ = fedexec.FedExecutionOps.evaluate_federation_model_on_existing_graph(tf_session=mon_sess,
																														  x_placeholders=_x_placeholders,
																														  y_placeholder=_y_placeholder,
																														  fedhost_eval_ops=fedhost_eval_ops,
																														  target_stat_name=TARGET_STAT_NAME,
																														  learner_id=cluster_spec,
																														  evaluate_validation_set=False)
							community_model_fedrounds_evaluation[federation_round_id+1] = test_eval_results['accuracy']
							print("\n\n")


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

	# ADAM configuration
	# model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.0015, momentum=0.0)

	# SGD With Momentum
	model_execution(tf_fedcluster=tf_fedcluster, learning_rate=0.05, momentum=0.0)