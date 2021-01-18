from experiments.tf_fedmodels.ner.ner_biocreative_fedmodel import NERBioCreativeFedModel
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection import imgdata_client
from utils.generic.json_ops import JSONUtil
from utils.devops.network_ops import NetOpsUtil
from utils.generic.time_ops import TimeUtil


import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import json
import os


scriptDirectory = os.path.dirname(os.path.realpath(__file__))
dirname = os.path.dirname(__file__)
HOSTNAME = NetOpsUtil.get_hostname()
print(HOSTNAME)

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ["GRPC_VERBOSITY"] = 'ERROR'

RUN_MULTI_LOCALHOST = True
NUM_CLUSTERS = 1
STARTING_PORT = 8221
USE_GPUS = True
LOGGING = False
COMPUTE_TEST_SET_STATS = True

# TRAINING HYPERPARAMETERS
LOCAL_BATCH_SIZE = 50
TRAINING_EXAMPLES_NUM = 400 # ALL 38343
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 150  # ALL 14078
IID_DATA_PARTITIONING = True

# FORMULA TO EMPLOY
SYNCHRONOUS_EXECUTION = True
DISCOUNTED_FEDERATED_VARIABLES_AVERAGING = True

# SYSTEM SIGNALS
TARGET_STAT_NAME = 'f1' # This can be the accuracy, precision, recall or f1
TARGET_STAT_SCORE = None  # Example: 0.95 - This is the target value of accuracy, precision, recall or f1-score
TARGET_EXEC_TIME_IN_MINS = 60  # Example: 80

if SYNCHRONOUS_EXECUTION:
	FEDERATION_ROUNDS = 1
	LOCAL_EPOCHS = 10
	# SEMI-SYNCHRONOUS POLICY UPDATE SIGNALS
	TARGET_LEARNERS = None
	TARGET_EPOCHS = None
else:
	# ASYNCHRONOUS EXECUTION
	FEDERATION_ROUNDS = 1
	LOCAL_EPOCHS = float("inf")
	# ASYNCHRONOUS POLICY UPDATE SIGNALS
	TARGET_LEARNERS = None
	TARGET_EPOCHS = 5


metis_logger.info('\n Federation Rounds: %s\n Local Batch Size: %s\n Local Epochs: %s\n Target Learners: %s\n Target Epochs: %s\n Target Stat Name-Score: %s-%s'
				% (FEDERATION_ROUNDS, LOCAL_BATCH_SIZE, LOCAL_EPOCHS, TARGET_LEARNERS, TARGET_EPOCHS, TARGET_STAT_NAME, TARGET_STAT_SCORE))

experiments_log_filename = "biocreative.fedrounds_{}.clients_{}.batchsize_{}.epochs_{}.policyspecs.discountedfedavg_{}.synchronous_{}.targetlearners_{}.targetepochs_{}.targetstat_{}.targetscore_{}.targetexectimemins_{}".format(
	FEDERATION_ROUNDS, NUM_CLUSTERS, LOCAL_BATCH_SIZE, LOCAL_EPOCHS, DISCOUNTED_FEDERATED_VARIABLES_AVERAGING, SYNCHRONOUS_EXECUTION,
	TARGET_LEARNERS, TARGET_EPOCHS, TARGET_STAT_NAME, TARGET_STAT_SCORE, TARGET_EXEC_TIME_IN_MINS
)
# experiments_log_filename = 'biocreative.standalone.execution.60mins.1xGPU'
experiments_log_filepath = scriptDirectory + "/../../resources/logs/testing_producibility/{}.json".format(experiments_log_filename)


def test_model_execution(tf_fedcluster, synchronized_execution=True, discounted_federated_variables_averaging=True):

	data_partitions_num = len(tf_fedcluster.fed_hosts)

	# Fill Data Cache, start DB Session
	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')
	# Set MetisDB session for grpc driver
	biocreativedb_session = imgdata_client.MetisDBSession(biocreative_session=True)
	biocreativedb_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM, dev_examples=DEV_EXAMPLES_NUM, test_examples=TEST_EXAMPLES_NUM)
	biocreativedb_session.partition_session_training_data(partitions_num=data_partitions_num, iid=IID_DATA_PARTITIONING)
	metis_logger.info('Data Partitioning Scheme: %s' % biocreativedb_session.partition_policy)
	metis_logger.info('Data Partitions: %s' % biocreativedb_session.partitions_num)
	metis_logger.info('Training Data Num: %s' % biocreativedb_session.train_examples_num)
	metis_logger.info('Testing Data Num: %s' % biocreativedb_session.test_examples_num)
	metis_logger.info(msg='Data Cache Filled')
	et = TimeUtil.current_milli_time()
	metis_logger.info('Data Cache Fill Time: %s ms' % TimeUtil.delta_diff_in_ms(et, st))

	nwords = biocreativedb_session.nwords
	nchars = biocreativedb_session.nchars
	ntags = biocreativedb_session.ntags
	tags_indices = [idx for idx, tag in enumerate(biocreativedb_session.vocab_tags) if tag.strip() != 'O']
	mnist_model = NERBioCreativeFedModel(nwords, nchars, ntags, tags_indices)
	federation_results = fedexec.FedExecutionOps.federated_between_graph_replication(fed_environment=tf_fedcluster,
																					 federation_model_obj=mnist_model,
																					 metis_db_session=biocreativedb_session,
																					 synchronous_execution=synchronized_execution,
																					 target_learners=TARGET_LEARNERS,
																					 target_epochs=TARGET_EPOCHS,
																					 session_target_stat_name=TARGET_STAT_NAME,
																					 target_stat_score=TARGET_STAT_SCORE,
																					 target_exec_time_mins=TARGET_EXEC_TIME_IN_MINS,
																					 compute_test_set_stats=COMPUTE_TEST_SET_STATS,
																					 logging=LOGGING)

	with open(experiments_log_filepath, 'w+') as fout:
		json.dump(obj=federation_results, fp=fout, indent=4, cls=JSONUtil.FedRoundsExecutionResultsJsonEncoder)
	print(federation_results)



if __name__ == "__main__":

	if RUN_MULTI_LOCALHOST:
		tf_fedcluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(hostname=HOSTNAME, clusters_num=NUM_CLUSTERS, federation_rounds=FEDERATION_ROUNDS,
																						batch_size_per_worker=LOCAL_BATCH_SIZE, epochs_per_worker=LOCAL_EPOCHS, workers_use_gpus=USE_GPUS,
																						starting_port=STARTING_PORT)

	else:
		# Cluster Setup
		SECURE_CLUSTER_SETTINGS_FILEPATH = '../../resources/config/tensorflow.federation.execution.10Learners.5Fast_atBDNF.5Slow_atLEARN.yaml'
		secure_cluster_settings_file = os.path.join(scriptDirectory, SECURE_CLUSTER_SETTINGS_FILEPATH)
		tf_fedcluster = fed_cluster_env.FedEnvironment.tf_federated_cluster_from_yaml(cluster_specs_file=secure_cluster_settings_file, init_cluster=False)

		# Override workers default local epochs and batch size
		tf_fedcluster.federation_rounds = FEDERATION_ROUNDS
		for fed_host in tf_fedcluster.fed_hosts:
			for fed_worker in fed_host.fed_worker_servers:
				fed_worker.local_epochs = LOCAL_EPOCHS
				fed_worker.batch_size = LOCAL_BATCH_SIZE

	test_model_execution(tf_fedcluster,
						 synchronized_execution=SYNCHRONOUS_EXECUTION,
						 discounted_federated_variables_averaging=DISCOUNTED_FEDERATED_VARIABLES_AVERAGING)