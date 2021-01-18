from experiments.tf_fedmodels.cnn.cnn2_mnist_model import MnistFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.json_ops import JSONUtil
from utils.generic.time_ops import TimeUtil

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import sys, getopt
import json
import yaml
import os

import numpy as np
np.random.seed(seed=1990)
import random
random.seed(a=1990)

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

# Execution Environment Variables
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ["GRPC_VERBOSITY"] = 'ERROR'
CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 10
SKEWNESS_FACTOR = float(os.environ["SKEWNESS_FACTOR"]) if "SKEWNESS_FACTOR" in os.environ else 1
BALANCED_RANDOM_PARTITIONING = str(os.environ['BALANCED_RANDOM_PARTITIONING'])=="True" if "BALANCED_RANDOM_PARTITIONING" in os.environ else False
BALANCED_CLASS_PARTITIONING = str(os.environ['BALANCED_CLASS_PARTITIONING'])=="True" if "BALANCED_CLASS_PARTITIONING" in os.environ else False
DATA_PARTITIONING_SCHEME = {'balanced_random_partitioning': BALANCED_RANDOM_PARTITIONING,
							'balanced_class_partitioning': BALANCED_CLASS_PARTITIONING,
							'skewness_factor': SKEWNESS_FACTOR,
							'classes_per_partition': CLASSES_PER_PARTITION}

metis_logger.info(msg="Federation Data Partitioning Scheme: {}".format(DATA_PARTITIONING_SCHEME))


EPOCH_TRAINING_LOGGING = False

# MULTI-LOCALHOST CONFIGS
RUN_MULTI_LOCALHOST_NUM_CLUSTERS = 10
RUN_MULTI_LOCALHOST_FEDERATION_ROUNDS = 100
RUN_MULTI_LOCALHOST_TARGET_LOCAL_EPOCHS = 4
RUN_MULTI_LOCALHOST_BATCH_SIZE_PER_WORKER = 50
RUN_MULTI_LOCALHOST_SERVERS_STARTING_PORT = 2222
RUN_MULTI_LOCALHOST_EXECUTION_TIME_IN_MINS = 180
RUN_MULTI_LOCALHOST_COMMUNITY_FUNCTION = "FedAvg"
RUN_MULTI_LOCALHOST_SYNCHRONOUS_EXECUTION = True
RUN_MULTI_LOCALHOST_VALIDATION_PERCENTAGE = 0
RUN_MULTI_LOCALHOST_VALIDATION_CYCLE_TOMBSTONES = 0
RUN_MULTI_LOCALHOST_VALIDATION_LOSS_PERCENTAGE_THRESHOLD = 0

# DATA SPECIFIC CONFIGS
TRAINING_EXAMPLES_NUM = 60000 # ALL: 50000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 10000 # ALL: 10000
TARGET_STAT_NAME = 'accuracy'
RUN_WITH_DISTORTED_IMAGES = True


def federation_model_execution(fed_environment, learning_rate, momentum):

	assert (isinstance(fed_environment, fed_cluster_env.FedEnvironment))
	data_partitions_num = len(fed_environment.fed_training_hosts)

	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')

	# Set MetisDB session for grpc driver
	metis_db_session = MetisDBSession(mnist_session=True,
									  working_directory="/nfs/isd/stripeli/metis_execution_tfrecords")
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM,
										  dev_examples=DEV_EXAMPLES_NUM,
										  test_examples=TEST_EXAMPLES_NUM,
										  distort_images=RUN_WITH_DISTORTED_IMAGES)
	metis_db_session.partition_session_training_data(partitions_num=data_partitions_num,
													 balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
													 balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
													 classes_per_partition=CLASSES_PER_PARTITION,
													 skewness_factor=SKEWNESS_FACTOR)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	metis_logger.info("Saving testing and training data of each partition as .TFRecords files")
	metis_db_session.save_testing_data_as_tfrecords()
	for pid in range(data_partitions_num):
		partition_fed_host_identifier = fed_environment.partition_idx_host_catalog.get(pid)
		partition_fed_host = [fed_host for fed_host in fed_environment.fed_training_hosts
							  if fed_host.host_identifier == partition_fed_host_identifier][0]
		partition_validation_proportion = partition_fed_host.fed_master.validation_proportion
		metis_db_session.save_partition_training_and_holdout_data_as_tfrecords(partition_id=pid,
																			   holdout_proportion=partition_validation_proportion)
	metis_logger.info("TF Records file saved")

	nnmodel = MnistFedModel(learning_rate=learning_rate, momentum=momentum)

	federation_results = fedexec.FedExecutionOps.federated_between_graph_replication(fed_environment=fed_environment,
																					 federation_model_obj=nnmodel,
																					 metis_db_session=metis_db_session,
																					 session_target_stat_name=TARGET_STAT_NAME,
																					 batch_level_log=EPOCH_TRAINING_LOGGING)
	metis_db_session.shutdown()
	return federation_results

if __name__ == "__main__":

	help_msg = 'cifar10_federated_main_sys_args.py -l <learning_rate> -m <momentum> -f <cluster_config_filepath>'
	argv = sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv, "hl:m:f", ["learning_rate=", "momentum=", "cluster_config_filepath="])
	except getopt.GetoptError:
		metis_logger.info(help_msg)
		sys.exit(2)

	model_arg1, model_arg2, model_arg3 = [False]*3
	RUN_MULTI_LOCALHOST = True
	for opt, arg in opts:
		if opt == '-h':
			metis_logger.info(help_msg)
			sys.exit(2)
		elif opt in ("-l", "--learning_rate"):
			learning_rate = float(arg)
			model_arg1 = True
		elif opt in ("-m", "--momentum"):
			momentum = float(arg)
			model_arg2 = True
		# elif opt in ("-b", "--batchsize"):
		# 	batchsize = int(arg)
		# 	model_arg3 = True

		if opt in ("-f", "--cluster_config_filepath"):
			CLUSTER_SETTINGS_FILEPATH = str(arg)
			metis_logger.info("Running Federation with specifications file: {}".format(CLUSTER_SETTINGS_FILEPATH))
			RUN_MULTI_LOCALHOST = False

	# Model Hyperparameters
	if all(x is True for x in [model_arg1, model_arg2]):
		LEARNING_RATE = learning_rate
		MOMENTUM = momentum
		# LOCAL_BATCH_SIZE = batchsize
		metis_logger.info("Learning rate is %s " % learning_rate)
		metis_logger.info("Momentum is %s " % momentum)
		# metis_logger.info("Batchsize is %s " % batchsize)
	else:
		metis_logger.info(msg="Not all the model hyperparameters are provided.")
		sys.exit(2)

	federation_execution_results = dict()
	if RUN_MULTI_LOCALHOST:
		metis_logger.info("Running Multi-Localhost since no cluster configuration file was provided.")
		fed_environment = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(clusters_num=RUN_MULTI_LOCALHOST_NUM_CLUSTERS,
																						  federation_rounds=RUN_MULTI_LOCALHOST_FEDERATION_ROUNDS,
																						  target_local_epochs=RUN_MULTI_LOCALHOST_TARGET_LOCAL_EPOCHS,
																						  batch_size_per_worker=RUN_MULTI_LOCALHOST_BATCH_SIZE_PER_WORKER,
																						  starting_port=RUN_MULTI_LOCALHOST_SERVERS_STARTING_PORT,
																						  execution_time_in_mins=RUN_MULTI_LOCALHOST_EXECUTION_TIME_IN_MINS,
																						  community_function=RUN_MULTI_LOCALHOST_COMMUNITY_FUNCTION,
																						  synchronous_execution=RUN_MULTI_LOCALHOST_SYNCHRONOUS_EXECUTION,
																						  validation_percentage=RUN_MULTI_LOCALHOST_VALIDATION_PERCENTAGE,
																						  validation_cycle_tombstones=RUN_MULTI_LOCALHOST_VALIDATION_CYCLE_TOMBSTONES,
																						  validation_cycle_loss_percentage_threshold=RUN_MULTI_LOCALHOST_VALIDATION_LOSS_PERCENTAGE_THRESHOLD)
		federation_rounds_results = federation_model_execution(fed_environment=fed_environment, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
		federation_execution_results["federation_rounds_results"] = federation_rounds_results
		federation_execution_results["federation_execution_configs"] = None


	else:

		fed_environment = fed_cluster_env.FedEnvironment.tf_federated_cluster_from_yaml(cluster_specs_file=CLUSTER_SETTINGS_FILEPATH, init_cluster=True)
		federation_rounds_results = federation_model_execution(fed_environment=fed_environment, learning_rate=LEARNING_RATE, momentum=MOMENTUM)
		fed_cluster_env.FedEnvironment.shutdown_yaml_created_tf_federated_cluster(fed_environment=fed_environment)
		federation_execution_configs = yaml.safe_load(open(CLUSTER_SETTINGS_FILEPATH, 'r'))
		federation_execution_configs["DataPartitioningScheme"] = DATA_PARTITIONING_SCHEME
		federation_execution_results["federation_rounds_results"] = federation_rounds_results
		federation_execution_results["federation_execution_configs"] = federation_execution_configs


	community_function = fed_environment.community_function
	batch_size = fed_environment.fed_training_hosts[0].fed_worker_servers[0].batch_size
	execution_time_in_mins = fed_environment.execution_time_in_mins
	ufrequencies = set([fed_host.fed_worker_servers[0].target_update_epochs for fed_host in fed_environment.fed_training_hosts])
	ufrequencies = '_'.join([str(uf) for uf in sorted(ufrequencies, reverse=True)])

	if fed_environment.synchronous_execution:
		experiments_log_filename = 'cifar10.classes_{}.BalancedRandom_{}.BalancedClass_{}.Skewness_{}.' \
								   'clients_10F.Function_Sync{}.RandomSeed_1990.SGDWithMomentum{}.learningrate_{}.batchsize_{}.' \
								   'synchronous_True.targetexectimemins_{}.UFREQUENCY={}.run_1' \
			.format(str(CLASSES_PER_PARTITION), str(BALANCED_RANDOM_PARTITIONING), str(BALANCED_CLASS_PARTITIONING), str(SKEWNESS_FACTOR).replace('.', 'p'),
					str(community_function), str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''),
					str(batch_size), str(execution_time_in_mins), str(ufrequencies))
	else:
		vloss = set([fed_host.fed_worker_servers[0].validation_cycle_loss_percentage_threshold for fed_host in fed_environment.fed_training_hosts])
		vloss = '_'.join([str(vl) for vl in sorted(vloss, reverse=True)])
		vtombs = set([fed_host.fed_worker_servers[0].validation_cycle_tombstones for fed_host in fed_environment.fed_training_hosts])
		vtombs = '_'.join([str(vt) for vt in sorted(vtombs, reverse=True)])
		experiments_log_filename = 'cifar10.classes_{}.BalancedRandom_{}.BalancedClass_{}.Skewness_{}.' \
								   'clients_10F.Function_Sync{}.VLoss_{}.Vtombs_{}.RandomSeed_1990.SGDWithMomentum{}.learningrate_{}.batchsize_{}.' \
								   'synchronous_False.targetexectimemins_{}.UFREQUENCY={}.run_1' \
				.format(str(CLASSES_PER_PARTITION), str(BALANCED_RANDOM_PARTITIONING), str(BALANCED_CLASS_PARTITIONING), str(SKEWNESS_FACTOR).replace('.', 'p'),
						str(community_function), str(vloss), str(vtombs), str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''), str(batch_size),
						str(execution_time_in_mins), str(ufrequencies))

	experiments_log_filepath = scriptDirectory + "/../../../../resources/logs/testing_producibility/{}.json".format(experiments_log_filename)
	metis_logger.info(msg="Writing output to file: {}".format(experiments_log_filepath))

	with open(experiments_log_filepath, 'w+') as fout:
		json.dump(obj=federation_execution_results, fp=fout, indent=4, cls=JSONUtil.FedRoundsExecutionResultsJsonEncoder)
