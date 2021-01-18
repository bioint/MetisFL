import argparse
import os
import random
import simplejson
import yaml

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import numpy as np
import tensorflow as tf

from experiments.tf_fedmodels.cnn.cnn2_cifar10_model import Cifar10FedModel
from metisdb.metisdb_catalog import MetisCatalog
from metisdb.metisdb_session import MetisDBSession
from metisdb.sqlalchemy_client import SQLAlchemyClient
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.objectdetection.imgdata_client import ImgDatasetLoader, ImgDatasetClient
from utils.generic.json_ops import JSONUtil
from utils.generic.time_ops import TimeUtil

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ["GRPC_VERBOSITY"] = 'ERROR'
EXEC_DIR = "/nfs/isd/stripeli/metis_tmp_exec_dir"

CLASSES_PER_PARTITION = int(os.environ["CLASSES_PER_PARTITION"]) if "CLASSES_PER_PARTITION" in os.environ else 10
SKEWNESS_FACTOR = float(os.environ["SKEWNESS_FACTOR"]) if "SKEWNESS_FACTOR" in os.environ else 0.0
BALANCED_RANDOM_PARTITIONING = str(os.environ["BALANCED_RANDOM_PARTITIONING"])=="True" \
	if "BALANCED_RANDOM_PARTITIONING" in os.environ else False
BALANCED_CLASS_PARTITIONING = str(os.environ["BALANCED_CLASS_PARTITIONING"])=="True" \
	if "BALANCED_CLASS_PARTITIONING" in os.environ else False
UNBALANCED_CLASS_PARTITIONING = str(os.environ["UNBALANCED_CLASS_PARTITIONING"])=="True" \
	if "UNBALANCED_CLASS_PARTITIONING" in os.environ else False
STRICTLY_UNBALANCED = str(os.environ["STRICTLY_UNBALANCED"])=="True" if "STRICTLY_UNBALANCED" in os.environ else False

# SEMI-SYNCHRONOUS SPECS
SEMI_SYNCHRONOUS_EXECUTION = str(os.environ["SEMI_SYNCHRONOUS_EXECUTION"])=="True" \
	if "SEMI_SYNCHRONOUS_EXECUTION" in os.environ else False
SEMI_SYNCHRONOUS_K_VALUE = float(os.environ["SEMI_SYNCHRONOUS_K_VALUE"]) \
	if "SEMI_SYNCHRONOUS_K_VALUE" in os.environ else 1
metis_logger.info("Semi-Sync: {}, K-Value:{} ".format(SEMI_SYNCHRONOUS_EXECUTION, SEMI_SYNCHRONOUS_K_VALUE))

DATA_PARTITIONING_SCHEME = {'balanced_random_partitioning': BALANCED_RANDOM_PARTITIONING,
							'balanced_class_partitioning': BALANCED_CLASS_PARTITIONING,
							'unbalanced_class_partitioning': UNBALANCED_CLASS_PARTITIONING,
							'strictly_unbalanced': STRICTLY_UNBALANCED,
							'skewness_factor': SKEWNESS_FACTOR,
							'classes_per_partition': CLASSES_PER_PARTITION}

metis_logger.info(msg="Federation Data Partitioning Scheme: {}".format(DATA_PARTITIONING_SCHEME))

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
sqlalchemy_client = SQLAlchemyClient(sqlite_instance_inmemory=True, maxconns=30)
METIS_CATALOG_CLIENT = MetisCatalog(sqlalchemy_client)

BATCH_LVL_LOGGING = False
RUN_WITH_DISTORTED_IMAGES = True

np.random.seed(seed=1990)
random.seed(a=1990)
tf.set_random_seed(seed=1990)

# Train the model
def federation_model_execution(fed_environment, learning_rate, momentum, num_corrupted_learners=0, corruption_level=0):

	assert (isinstance(fed_environment, fed_cluster_env.FedEnvironment))

	img_dataset_loader = ImgDatasetLoader(cifar10_loader=True, train_examples=50000, test_examples=10000)
	img_dataset_loader.load_image_datasets()
	# img_dataset_loader.partition_training_data(partitions_num=len(fed_environment.fed_training_hosts),
	# 										   balanced_random_partitioning=BALANCED_RANDOM_PARTITIONING,
	# 										   balanced_class_partitioning=BALANCED_CLASS_PARTITIONING,
	# 										   unbalanced_class_partitioning=UNBALANCED_CLASS_PARTITIONING,
	# 										   skewness_factor=SKEWNESS_FACTOR,
	# 										   strictly_unbalanced=STRICTLY_UNBALANCED,
	# 										   classes_per_partition=CLASSES_PER_PARTITION)
	img_dataset_loader.partition_training_data(partitions_num=len(fed_environment.fed_training_hosts),
											   MLSYS_REBUTTAL_REVIEWER2=True)
	total_corrupted_learners = 0
	img_dataset_clients = list()
	for training_host_idx, training_host in enumerate(fed_environment.fed_training_hosts):

		learner_id = training_host.host_identifier
		validation_percentage = training_host.fed_worker_servers[0].validation_proportion
		corrupt_current_learner = True if total_corrupted_learners < num_corrupted_learners else False
		corrupt_labels_uniform = True if corruption_level == 1 and corrupt_current_learner else False
		corrupt_images_opposite = True if corruption_level == 2 and corrupt_current_learner else False
		img_dataset_client = ImgDatasetClient(image_dataset_loader=img_dataset_loader, learner_id=learner_id,
											  learner_partition_idx=training_host_idx,
											  validation_percentage=validation_percentage,
											  distort_training_images=RUN_WITH_DISTORTED_IMAGES, cifar10_client=True,
											  corrupt_labels_uniform=corrupt_labels_uniform,
											  corrupt_images_opposite=corrupt_images_opposite)
		img_dataset_clients.append(img_dataset_client)

		if corrupt_current_learner:
			total_corrupted_learners += 1

	metis_db_session = MetisDBSession(METIS_CATALOG_CLIENT, img_dataset_clients, is_classification=True,
									  num_classes=img_dataset_loader.num_classes,
									  negative_classes_indices=img_dataset_loader.negative_classes_indices,
									  is_eval_output_scalar=img_dataset_loader.is_eval_output_single_scalar)

	for img_dataset_client in img_dataset_clients:
		current_timestamp = str(int(TimeUtil.current_milli_time()))
		tfrecord_training_filepath = os.path.join(EXEC_DIR, "cifar10.train.tfrecord.{}.ts_{}"\
			.format(img_dataset_client.learner_partition_idx, current_timestamp))
		tfrecord_testing_filepath = os.path.join(EXEC_DIR, "cifar10.test.tfrecord.{}.ts_{}" \
			.format(img_dataset_client.learner_partition_idx, current_timestamp))
		if img_dataset_client.learner_validation_data is not None:
			tfrecord_validation_filepath = os.path.join(EXEC_DIR, "cifar10.validation.tfrecord.{}.ts_{}" \
				.format(img_dataset_client.learner_partition_idx, current_timestamp))
		else:
			tfrecord_validation_filepath = None

		training_data_size, training_tfrecords_schema = img_dataset_client.generate_tfrecords(
			tfrecord_output_filename=tfrecord_training_filepath, is_training=True)

		validation_data_size, validation_tfrecords_schema = img_dataset_client.generate_tfrecords(
			tfrecord_output_filename=tfrecord_validation_filepath, is_validation=True)

		testing_data_size, testing_tfrecords_schema = img_dataset_client.generate_tfrecords(
			tfrecord_output_filename=tfrecord_testing_filepath, is_testing=True)

		metis_db_session.register_tfrecords_volumes(img_dataset_client.learner_id, tfrecord_training_filepath,
													tfrecord_validation_filepath, tfrecord_testing_filepath)
		metis_db_session.register_tfrecords_schemata(img_dataset_client.learner_id, training_tfrecords_schema,
													 validation_tfrecords_schema, testing_tfrecords_schema)
		metis_db_session.register_tfrecords_examples_num(img_dataset_client.learner_id, training_data_size,
														 validation_data_size, testing_data_size)
		metis_db_session.register_tfdatasets_schemata(img_dataset_client.learner_id)

	# Once the data partitioning and the learners registration take place, save the federation data distribution.
	federation_data_distribution = img_dataset_loader.retrieve_all_partitions_training_dataset_stats(
		to_json_representation=True, partitions_labels=[host.host_identifier
														for host in fed_environment.fed_training_hosts])

	# Run federation.
	nnmodel = Cifar10FedModel(learning_rate=learning_rate, momentum=momentum,
							  run_with_distorted_images=RUN_WITH_DISTORTED_IMAGES)
	federation_rounds = fedexec.FedExecutionOps.federated_between_graph_replication(fed_environment=fed_environment,
																					federation_model_obj=nnmodel,
																					metis_db_session=metis_db_session,
																					session_target_stat_name='accuracy',
																					batch_level_log=BATCH_LVL_LOGGING)
	# Delete all tf records of current session.
	for img_dataset_client in img_dataset_clients:
		train_tfrecord, valid_tfrecord, test_tfrecord = metis_db_session.get_tfrecords_volumes(
			img_dataset_client.learner_id,
			True, True, True)
		os.remove(train_tfrecord)
		os.remove(test_tfrecord)
		if valid_tfrecord is not None:
			os.remove(valid_tfrecord)
	metis_db_session.shutdown()

	# Create final json object holding federation execution metadata.
	federation_results = dict()
	federation_results['federation_rounds'] = federation_rounds
	federation_results['federation_data_distribution'] = federation_data_distribution
	return federation_results


def main(args):

	fed_environment = fed_cluster_env.FedEnvironment.tf_federated_cluster_from_yaml(
		cluster_specs_file=args.federation_environment_filepath,
		init_cluster=True)
	federation_results = federation_model_execution(fed_environment, float(args.learning_rate), float(args.momentum),
													int(args.num_corrupted_learners), int(args.corruption_level))

	federation_execution_results = dict()
	federation_environment = yaml.safe_load(open(args.federation_environment_filepath))

	federation_execution_results["federation_environment"] = federation_environment
	federation_execution_results["federation_results"] = federation_results

	community_function = fed_environment.community_function
	batch_size = fed_environment.fed_training_hosts[0].fed_worker_servers[0].batch_size
	execution_time_in_mins = fed_environment.execution_time_in_mins

	experiments_log_filename = 'cifar10.classes_{}.BalancedRandom_{}.BalancedClass_{}.NISU_{}.Skewness_{}.' \
							   'clients_10F.Function_{}.SGDWithMomentum{}.lr_{}.b_{}.' \
							   '{}.targetexectimemins_{}.UFREQUENCY={}.corrupted_{}.run_1'
	# experiments_log_filename = 'cifar10.classes_{}.BalancedRandom_{}.BalancedClass_{}.NISU_{}.Skewness_{}.' \
	# 						   'clients_10F.Function_Sync{}.FedProx.m0001.lr_{}.b_{}.' \
	# 						   '{}.targetexectimemins_{}.UFREQUENCY={}.run_1'
	if SEMI_SYNCHRONOUS_EXECUTION:
		ufrequencies = 0
		SEMI_SYNCHRONOUS_K_VALUE_STR = str(SEMI_SYNCHRONOUS_K_VALUE).replace('.','')
		synchronization_policy_text = 'semisynchronous_True.k_{}'.format(SEMI_SYNCHRONOUS_K_VALUE_STR)
	else:
		ufrequencies = set([fed_host.fed_worker_servers[0].target_update_epochs
							for fed_host in fed_environment.fed_training_hosts])
		ufrequencies = '_'.join([str(uf) for uf in sorted(ufrequencies, reverse=True)])
		if fed_environment.synchronous_execution:
			synchronization_policy_text = 'synchronous_True'
		else:
			synchronization_policy_text = 'synchronous_False'

	# experiments_log_filename = experiments_log_filename.format(
	# 	str(CLASSES_PER_PARTITION), str(BALANCED_RANDOM_PARTITIONING), str(BALANCED_CLASS_PARTITIONING),
	# 	str(STRICTLY_UNBALANCED), str(SKEWNESS_FACTOR).replace('.', 'p'), str(community_function),
	# 	str(args.momentum).replace('.', ''), str(args.learning_rate).replace('.', ''), str(batch_size),
	# 	synchronization_policy_text, str(execution_time_in_mins), str(ufrequencies))
	experiments_log_filename = experiments_log_filename.format(
		str(CLASSES_PER_PARTITION), str(BALANCED_RANDOM_PARTITIONING), str(BALANCED_CLASS_PARTITIONING),
		str(STRICTLY_UNBALANCED), str(SKEWNESS_FACTOR).replace('.', 'p'), str(community_function),
		str(args.momentum).replace('.', ''), str(args.learning_rate).replace('.', ''), str(batch_size),
		synchronization_policy_text, str(execution_time_in_mins), str(ufrequencies), str(args.num_corrupted_learners))

	experiments_log_filepath = scriptDirectory + "/../../../execution_logs/{}.json"\
		.format(experiments_log_filename)
	with open(experiments_log_filepath, 'w+') as fout:
		simplejson.dump(obj=federation_execution_results, fp=fout, indent=4, ignore_nan=True,
						cls=JSONUtil.FedRoundsExecutionResultsJsonEncoder)
	fed_environment.shutdown_yaml_created_tf_federated_cluster()


def get_args():
	parser = argparse.ArgumentParser()

	# Required params
	parser.add_argument("--federation_environment_filepath", help="path to training data")
	parser.add_argument("--learning_rate", help="learning rate value", default=0.05)
	parser.add_argument("--momentum", help="momentum value", default=0.75)
	parser.add_argument("--num_corrupted_learners", help="number of corrupted learners", default=0)
	parser.add_argument("--corruption_level", help="For uniformly corrupted labels set 1.\n"
												   "For the negative/opposite of input image set 2.", default=0)

	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	main(args)
	METIS_CATALOG_CLIENT.shutdown()
