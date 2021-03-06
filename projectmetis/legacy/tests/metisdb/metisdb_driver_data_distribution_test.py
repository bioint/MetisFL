from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import MetisDatasetClient
from utils.generic.json_ops import JSONUtil
from utils.generic.time_ops import TimeUtil
from collections import defaultdict

import federation.fed_cluster_env as fed_cluster_env
import numpy as np
import random
import json
import os

random.seed(1990)
np.random.seed(1990)

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

scriptDirectory = os.path.dirname(os.path.realpath(__file__))

TRAINING_EXAMPLES_NUM = 697932 #697932 #50000 #697932
DEV_EXAMPLES_NUM = 0
TEST_EXAMPLES_NUM = 0 #116323

PARTITIONS_NUM = 10
CLASSES_PER_PARTITION = 30
SKEWNESS_FACTOR = 0.0

# DATA_DISTRIBUTION_OUTPUT_FILE = os.path.join(scriptDirectory, "cifar100.classes_{}.Skewness_{}.Skewed.DataDistribution.json")
DATA_DISTRIBUTION_OUTPUT_FILE = os.path.join(scriptDirectory, "extended_mnist.classes_{}.Skewness_{}.Skewed.DataDistribution.json")

DATA_DISTRIBUTION_OUTPUT_FILE = DATA_DISTRIBUTION_OUTPUT_FILE.format(CLASSES_PER_PARTITION, "p".join(str(SKEWNESS_FACTOR).split(".")))
print(DATA_DISTRIBUTION_OUTPUT_FILE)

if __name__=="__main__":

	fed_cluster = fed_cluster_env.FedEnvironment.init_multi_localhost_tf_clusters(clusters_num=PARTITIONS_NUM,
																				  starting_port=1111)

	# Fill Data Cache, start DB Session
	st = TimeUtil.current_milli_time()
	metis_logger.info(msg='Initializing Data Cache...')

	metis_db_session = MetisDatasetClient(working_directory="/nfs/isd/stripeli/metis_execution_tfrecords",
									  extended_mnist_session=True,
									  extended_mnist_byclass=True)
	# metis_db_session = MetisDBSession(working_directory="/nfs/isd/stripeli/metis_execution_tfrecords",
	# 								  cifar100_session=True)
	# metis_db_session = MetisDBSession(working_directory="/nfs/isd/stripeli/metis_execution_tfrecords",
	# 								  cifar10_session=True)

	metis_db_session.load_session_dataset(train_examples=TRAINING_EXAMPLES_NUM,
										  dev_examples=DEV_EXAMPLES_NUM,
										  test_examples=TEST_EXAMPLES_NUM,
										  distort_images=True)
	metis_db_session.partition_session_training_data(partitions_num=PARTITIONS_NUM,
													 balanced_random_partitioning=False,
													 balanced_class_partitioning=True,
													 unbalanced_class_partitioning=False,
													 strictly_unbalanced=False,
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

	# # Write data distribution to a text file
	# with open('data_file.csv', 'w', newline='\n') as fout:
	# 	writer = csv.writer(fout, delimiter="|")
	# 	writer.writerow(["partition_id", "x_dtype", "y_dtype", "x_value", "y_value"])
	# 	for partition_id, partition_data in metis_db_session.partitioned_training_data.items():
	# 		partition_input = partition_data.input
	# 		partition_output = partition_data.output
	# 		for training_sample in zip(partition_input, partition_output):
	# 			training_sample_x = training_sample[0]
	# 			training_sample_x_dtype = training_sample_x.dtype
	# 			training_sample_x = ','.join([str(x) for x in training_sample_x.tolist()])
	# 			training_sample_y = training_sample[1]
	# 			training_sample_y_dtype = training_sample_y.dtype
	#
	# 			row_record = [partition_id, training_sample_x_dtype, training_sample_y_dtype, training_sample_x, training_sample_y]
	# 			writer.writerow(row_record)

	metis_db_session.shutdown()


	federation_data_distribution = dict()
	partitions_dataset_stats = metis_db_session.retrieve_all_partitions_datasets_stats(to_json_representation=True)
	for pid, partition_dataset_stats in partitions_dataset_stats.items():
		host_id = fed_cluster.partition_idx_host_catalog.get(pid)
		federation_data_distribution[host_id] = partition_dataset_stats

	federation_results = defaultdict(dict)
	federation_results["federation_round_0"]["federation_data_distribution"] = federation_data_distribution

	federation_results["federation_round_0"]["hosts_results"] = defaultdict(dict)
	for fed_host in fed_cluster.fed_training_hosts:
		federation_results["federation_round_0"]["hosts_results"][fed_host.host_identifier]["host_id"] = fed_host.host_identifier
		federation_results["federation_round_0"]["hosts_results"][fed_host.host_identifier]["training_devices"] = fed_host.host_training_devices

	with open(DATA_DISTRIBUTION_OUTPUT_FILE, 'w+') as fout:
		json.dump(obj=federation_results, fp=fout, indent=4, cls=JSONUtil.FedRoundsExecutionResultsJsonEncoder)