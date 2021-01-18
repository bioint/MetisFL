from experiments.tf_fedmodels.cnn.cnn2_fmnist_model import FmnistFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDBSession
from utils.generic.json_ops import JSONUtil
from utils.generic.time_ops import TimeUtil

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import sys, getopt
import json
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

# NOTE The following implementation of mnist_deep_CNN is largely based on: https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/mnist/mnist_deep.py
# tf.logging.set_verbosity(4)  # Enable INFO verbosity

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ["GRPC_VERBOSITY"] = 'ERROR'
MIN_VLOSS = float(os.environ['MINVLOSS']) if 'MINVLOSS' in os.environ else 1.0
MAX_VLOSS = float(os.environ['MAXVLOSS']) if 'MAXVLOSS' in os.environ else 5.0
CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 10
BALANCED_PARTITIONING=str(os.environ['BALANCED_PARTITIONING'])=="True" if 'BALANCED_PARTITIONING' in os.environ else True

RUN_MULTI_LOCALHOST = False
NUM_CLUSTERS = 10
USE_GPUS = True
LOGGING = False
COMPUTE_TEST_SET_STATS = True

TRAINING_EXAMPLES_NUM = 60000 # ALL 60000
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 10000  # ALL 10000
SHARDS_NUM = None

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'accuracy'
TARGET_STAT_SCORE = None # Example: 0.95


def federation_model_execution(tf_fedcluster, learning_rate, momentum, controller_host_port="bdnf.isi.edu:50051",
							   evaluator_host_port="bdnf.isi.edu:8787", validation_proportion=0.05,
							   synchronous_execution=True, execution_time=80):

	assert (isinstance(tf_fedcluster, fed_cluster_env.FedEnvironment))
	data_partitions_num = len(tf_fedcluster.fed_training_hosts)

	# Fill Data Cache, start DB Session
	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	# Set MetisDB session for federation driver
	metis_db_session = MetisDBSession(fmnist_session=True)
	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM, dev_examples=DEV_EXAMPLES_NUM, test_examples=TEST_EXAMPLES_NUM)
	metis_db_session.partition_session_training_data(partitions_num=data_partitions_num, classes_per_partition=CLASSES_PER_PARTITION, balanced=BALANCED_PARTITIONING, skewness_factor=1.5)
	metis_logger.info('Data Partitioning Scheme: %s' % metis_db_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % metis_db_session.partitions_num)
	metis_logger.info('Classes Per Partition: %s' % CLASSES_PER_PARTITION)
	metis_logger.info('Training Data Num: %s' % metis_db_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % metis_db_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nnmodel = FmnistFedModel(learning_rate=learning_rate, momentum=momentum)

	federation_results = fedexec.FedExecutionOps.federated_between_graph_replication(fed_environment=tf_fedcluster,
																					 federation_model_obj=nnmodel,
																					 metis_db_session=metis_db_session,
																					 synchronous_execution=synchronous_execution,
																					 session_target_stat_name=TARGET_STAT_NAME,
																					 compute_test_set_stats=COMPUTE_TEST_SET_STATS,
																					 target_exec_time_mins=execution_time,
																					 controller_host_port=controller_host_port,
																					 evaluator_host_port=evaluator_host_port,
																					 validation_proportion=validation_proportion,
																					 logging=LOGGING)

	with open(experiments_log_filepath, 'w+') as fout:
		json.dump(obj=federation_results, fp=fout, indent=4, cls=JSONUtil.FedRoundsExecutionResultsJsonEncoder)


if __name__ == "__main__":

	help_msg = 'fmnist_federated_main_sys_args.py -s <synchronous> -l <learning_rate> -m <momentum> -b <batchsize> -u <update_frequency> -r <federation_rounds>  -t <execution_time>'
	argv = sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv, "hs:l:m:b:u:r:t:", ["synchronous=", "learning_rate=", "momentum=", "batchsize=", "update_frequency=", "federation_rounds=", "execution_time="])
	except getopt.GetoptError:
		metis_logger.info(help_msg)
		sys.exit(2)

	synchronous = None
	ufrequency = None
	execution_time = None
	federation_rounds = None
	TARGET_EXEC_TIME_IN_MINS = None
	model_arg1, model_arg2, model_arg3 = [False]*3
	for opt, arg in opts:
		if opt == '-h':
			metis_logger.info(help_msg)
			sys.exit(2)
		if opt in ("-s", "--synchronous"):
			synchronous = arg == "True"
		elif opt in ("-l", "--learning_rate"):
			learning_rate = float(arg)
			model_arg1 = True
		elif opt in ("-m", "--momentum"):
			momentum = float(arg)
			model_arg2 = True
		elif opt in ("-b", "--batchsize"):
			batchsize = int(arg)
			model_arg3 = True
		elif opt in ("-u", "--update_frequency"):
			ufrequency = int(arg)
		elif opt in ("-r", "--federation_rounds"):
			federation_rounds = int(arg)
		elif opt in ("-t", "--execution_time"):
			execution_time = int(arg)
			sys_arg4 = True

	# Model Hyperparameters
	if all(x is True for x in [model_arg1, model_arg2, model_arg3]):
		LEARNING_RATE = learning_rate
		MOMENTUM = momentum
		LOCAL_BATCH_SIZE = batchsize
		metis_logger.info("Learning rate is %s " % learning_rate)
		metis_logger.info("Momentum is %s " % momentum)
		metis_logger.info("Batchsize is %s " % batchsize)
	else:
		metis_logger.info(msg="Not all the model hyperparameters are provided.")
		sys.exit(2)

	# System Hyperparameters
	if synchronous is None and ufrequency is None and execution_time is None:
		metis_logger.info("Synchronous or Asynchronous execution mode, Update Frequency and Execution Time must be provided")
		sys.exit(2)
	else:
		SYNCHRONOUS_EXECUTION = synchronous
		UPDATE_FREQUENCY_EPOCHS = ufrequency
		TARGET_EXEC_TIME_IN_MINS = execution_time

		metis_logger.info("Update Frequency is %s " % UPDATE_FREQUENCY_EPOCHS)
		if UPDATE_FREQUENCY_EPOCHS >= 1:
			SERVERS_STARTING_PORT = 4222
			CONTROLLER_HOST_PORT = "bdnf.isi.edu:40051"
		else:
			SERVERS_STARTING_PORT = 5222
			CONTROLLER_HOST_PORT = "bdnf.isi.edu:50051"

		if SYNCHRONOUS_EXECUTION:
			if UPDATE_FREQUENCY_EPOCHS == 0:
				metis_logger.info(msg="During synchronous execution we need to execute at least 1 epoch.")
				sys.exit(2)
			if federation_rounds is None:
				metis_logger.info(msg="Synchronous execution mode requires the number of federation rounds.")
				sys.exit(2)
			else:
				FEDERATION_ROUNDS = federation_rounds
			LOCAL_EPOCHS = UPDATE_FREQUENCY_EPOCHS
			# SEMI-SYNCHRONOUS POLICY UPDATE SIGNALS
			TARGET_LEARNERS = None
		else:
			FEDERATION_ROUNDS = 1
			LOCAL_EPOCHS = float("inf")
			# ASYNCHRONOUS POLICY LEARNERS NUM UPDATE SIGNAL
			TARGET_LEARNERS = None

	MIN_VLOSS = float(os.environ['MINVLOSS']) if 'MINVLOSS' in os.environ else 1.0
	MAX_VLOSS = float(os.environ['MAXVLOSS']) if 'MAXVLOSS' in os.environ else 5.0
	if MIN_VLOSS.is_integer():
		MIN_VLOSS = int(MIN_VLOSS)
	if MAX_VLOSS.is_integer():
		MAX_VLOSS = int(MAX_VLOSS)

	# experiments_log_filename = 'fmnist.classes_{}.clients_5F5S.Function_SyncFedValidation005WithLoss.AdamOpt.learningrate_{}.batchsize_{}.synchronous_{}.targetexectimemins_{}.UFREQUENCY={}.run_1' \
	# 	.format(str(CLASSES_PER_PARTITION), str(LEARNING_RATE).replace('.', ''), str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''), str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))
	# experiments_log_filename = 'fmnist.classes_{}.centralized.execution.300epochs.60000examples.1xGPU.SGDWithMomentum.learningrate_{}.batchsize_{}.run_1' \
	# 	.format(str(CLASSES_PER_PARTITION), str(LEARNING_RATE).replace('.', ''), str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''), str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))

	if UPDATE_FREQUENCY_EPOCHS > 0:

		if BALANCED_PARTITIONING:
			experiments_log_filename = 'fmnist.classes_{}.clients_5F5S.Function_SyncFedAvg.RandomSeed_1990.SGDWithMomentum{}.learningrate_{}.batchsize_{}.synchronous_{}.targetexectimemins_{}.UFREQUENCY={}.run_1' \
				.format(str(CLASSES_PER_PARTITION), str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''),
						str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''),
						str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))
		else:
			experiments_log_filename = 'fmnist.classes_{}.clients_5F5S.UnbalancedDataDistrib.Skewness_1p5.Function_SyncFedAvg.RandomSeed_1990.SGDWithMomentum{}.learningrate_{}.batchsize_{}.synchronous_{}.targetexectimemins_{}.UFREQUENCY={}.run_1' \
				.format(str(CLASSES_PER_PARTITION), str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''),
						str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''),
						str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))
	else:
		# VPT: Learners Validation Phase Tombstone
		# VCT: Learners Validation Cycle Tombstone
		FAST_LEARNER_VC_LOSS_PCT_THRESHOLD = float(os.environ["FAST_LEARNER_VC_LOSS_PCT_THRESHOLD"])
		if FAST_LEARNER_VC_LOSS_PCT_THRESHOLD.is_integer():
			FAST_LEARNER_VC_LOSS_PCT_THRESHOLD = int(FAST_LEARNER_VC_LOSS_PCT_THRESHOLD)
		SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD = float(os.environ["SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD"])
		if SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD.is_integer():
			SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD = int(SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD)

		if BALANCED_PARTITIONING:
			experiments_log_filename = 'fmnist.classes_{}.clients_5F5S.Function_AsyncFedVF1MacroSmooth_WithVal_{}.StalenessCriterionAtVC20.VCTomb_{}_{}.VCLossThresh_{}_{}.SGDWithMomentum{}.learningrate_{}.batchsize_{}.synchronous_{}.targetexectimemins_{}.UFREQUENCY={}.run_1' \
				.format(str(CLASSES_PER_PARTITION), str(float(os.environ["VALIDATION_PROPORTION"])).replace('.',''),
						str(int(os.environ["FAST_LEARNER_VC_TOMBSTONES"])), str(int(os.environ["SLOW_LEARNER_VC_TOMBSTONES"])),
						str(FAST_LEARNER_VC_LOSS_PCT_THRESHOLD).replace(".", ""), str(SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD).replace(".", ""),
						str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''),
						str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''),
						str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))
		else:
			experiments_log_filename = 'fmnist.classes_{}.clients_5F5S.UnbalancedDataDistrib.Skewness_1p5.Function_AsyncFedVF1MacroSmooth_WithVal_{}.VCTomb_{}_{}.VCLossThresh_{}_{}.SGDWithMomentum{}.learningrate_{}.batchsize_{}.synchronous_{}.targetexectimemins_{}.UFREQUENCY={}.run_1' \
				.format(str(CLASSES_PER_PARTITION), str(float(os.environ["VALIDATION_PROPORTION"])).replace('.',''),
						str(int(os.environ["FAST_LEARNER_VC_TOMBSTONES"])), str(int(os.environ["SLOW_LEARNER_VC_TOMBSTONES"])),
						str(FAST_LEARNER_VC_LOSS_PCT_THRESHOLD).replace(".", ""), str(SLOW_LEARNER_VC_LOSS_PCT_THRESHOLD).replace(".", ""),
						str(MOMENTUM).replace('.', ''), str(LEARNING_RATE).replace('.', ''),
						str(LOCAL_BATCH_SIZE).replace('.', ''), str(synchronous), str(TARGET_EXEC_TIME_IN_MINS).replace('.', ''),
						str(UPDATE_FREQUENCY_EPOCHS).replace('.', ''))

	experiments_log_filepath = scriptDirectory + "/../../../resources/logs/testing_producibility/{}.json".format(experiments_log_filename)
	open(experiments_log_filepath, "w+")
	metis_logger.info(msg="Writing output to file: {}".format(experiments_log_filepath))


	if RUN_MULTI_LOCALHOST:

		tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(hostname="bdnf.isi.edu",
																						clusters_num=NUM_CLUSTERS,
																						federation_rounds=FEDERATION_ROUNDS,
																						target_local_epochs=UPDATE_FREQUENCY_EPOCHS,
																						batch_size_per_worker=LOCAL_BATCH_SIZE,
																						starting_port=SERVERS_STARTING_PORT)

	else:

		SECURE_CLUSTER_SETTINGS_FILEPATH = '../../../resources/config/tensorflow.federation.execution.10Learners.5Fast_atBDNF.5Slow_atLEARN.yaml'
		# CLUSTER_SETTINGS_FILE_TEMPLATE = '../../../resources/config/tensorflow.federation.execution.10Learners.10Fast_atBDNF.yaml'
		secure_cluster_settings_file = os.path.join(scriptDirectory, SECURE_CLUSTER_SETTINGS_FILEPATH)
		tf_fedcluster = fed_cluster_env.FedEnvironment.tf_federated_cluster_from_yaml(cluster_specs_file=secure_cluster_settings_file, init_cluster=False)

		# Override federation rounds, local epochs and batch size
		tf_fedcluster.federation_rounds = FEDERATION_ROUNDS
		for fed_host in tf_fedcluster.fed_hosts:
			for fed_worker in fed_host.fed_worker_servers:

				fed_worker.local_epochs = LOCAL_EPOCHS
				if LOCAL_BATCH_SIZE is not None:
					fed_worker.local_epochs = LOCAL_EPOCHS
					fed_worker.batch_size = LOCAL_BATCH_SIZE
					fed_worker.target_update_epochs = UPDATE_FREQUENCY_EPOCHS


	federation_model_execution(tf_fedcluster=tf_fedcluster,
							   learning_rate=LEARNING_RATE,
							   momentum=MOMENTUM,
							   synchronous_execution=SYNCHRONOUS_EXECUTION,
							   execution_time=TARGET_EXEC_TIME_IN_MINS,
							   controller_host_port=CONTROLLER_HOST_PORT)
