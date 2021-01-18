import argparse
import os
import random
import simplejson
import yaml

import federation.fed_cluster_env as fed_cluster_env
import federation.fed_execution as fedexec
import numpy as np
import tensorflow as tf

from collections import defaultdict
from experiments.tf_fedmodels.cnn.cnn5_brain_age import BrainAgeCNNFedModel
from metisdb.metisdb_catalog import MetisCatalog
from metisdb.metisdb_session import MetisDBSession
from metisdb.sqlalchemy_client import SQLAlchemyClient
from utils.brainage.neurodata_client import NeuroDatasetClient
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.generic.json_ops import JSONUtil
from utils.generic.time_ops import TimeUtil

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ["GRPC_VERBOSITY"] = 'ERROR'
os.path.dirname(os.path.realpath(__file__))

# SEMI-SYNCHRONOUS SPECS
SEMI_SYNCHRONOUS_EXECUTION = str(os.environ["SEMI_SYNCHRONOUS_EXECUTION"])=="True" \
	if "SEMI_SYNCHRONOUS_EXECUTION" in os.environ else False
SEMI_SYNCHRONOUS_K_VALUE = float(os.environ["SEMI_SYNCHRONOUS_K_VALUE"]) \
	if "SEMI_SYNCHRONOUS_K_VALUE" in os.environ else 1
metis_logger.info("Semi-Sync: {}, K-Value:{} ".format(SEMI_SYNCHRONOUS_EXECUTION, SEMI_SYNCHRONOUS_K_VALUE))

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
sqlalchemy_client = SQLAlchemyClient(postgres_instance=True, maxconns=100)
METIS_CATALOG_CLIENT = MetisCatalog(sqlalchemy_client)

BATCH_LVL_LOGGING = False

np.random.seed(seed=1990)
random.seed(a=1990)
tf.set_random_seed(seed=1990)

# Train the model
def federation_model_execution(fed_environment):

	assert (isinstance(fed_environment, fed_cluster_env.FedEnvironment))

	neurodata_clients = list()
	data_task_column = 'age_at_scan'
	data_volume_column = '9dof_2mm_vol' # OTHERS: 'vol_path'

	for training_host in fed_environment.fed_training_hosts:

		learner_id = training_host.host_identifier
		neurodata_client = NeuroDatasetClient(learner_id=learner_id, rows=91, cols=109, depth=91)

		training_filepath = training_host.dataset_configs.train_dataset_mappings
		validation_filepath = training_host.dataset_configs.validation_dataset_mappings
		testing_filepath = training_host.dataset_configs.test_dataset_mappings

		neurodata_client.parse_data_mappings_file(
			filepath=training_filepath,
			data_volume_column=data_volume_column,
			csv_reader_schema={data_task_column: np.float32, data_volume_column: np.str},
			is_training=True)

		# Validation datasets are not required thus we check whether they are defined or not!
		if validation_filepath is not None and validation_filepath != "":
			neurodata_client.parse_data_mappings_file(
				filepath=training_host.dataset_configs.validation_dataset_mappings,
				data_volume_column=data_volume_column,
				csv_reader_schema={data_task_column: np.float32, data_volume_column: np.str},
				is_validation=True)

		neurodata_client.parse_data_mappings_file(
			filepath=testing_filepath,
			data_volume_column=data_volume_column,
			csv_reader_schema={data_task_column: np.float32, data_volume_column: np.str},
			is_testing=True)

		neurodata_clients.append(neurodata_client)

	metis_db_session = MetisDBSession(METIS_CATALOG_CLIENT, neurodata_clients, is_regression=True)

	for idx, neurodata_client in enumerate(neurodata_clients):
		tfrecord_training_filepath = "/lfs1/stripeli/condaprojects/federatedneuroimaging/tmp_exec_dir/neuro.train.tfrecord.{}".format(idx)
		tfrecord_testing_filepath = "/lfs1/stripeli/condaprojects/federatedneuroimaging/tmp_exec_dir/neuro.test.tfrecord.{}".format(idx)
		tfrecord_validation_filepath = "/lfs1/stripeli/condaprojects/federatedneuroimaging/tmp_exec_dir/neuro.validation.tfrecord.{}".format(idx) \
			if neurodata_client.client_has_validation_data() else None

		training_data_size, training_tfrecords_schema = neurodata_client.generate_tfrecords(
		  data_volume_column=data_volume_column, tfrecord_output_filename=tfrecord_training_filepath, is_training=True)

		validation_data_size, validation_tfrecords_schema = neurodata_client.generate_tfrecords(
			data_volume_column=data_volume_column, tfrecord_output_filename=tfrecord_validation_filepath,
			is_validation=True)

		testing_data_size, testing_tfrecords_schema = neurodata_client.generate_tfrecords(
			data_volume_column=data_volume_column, tfrecord_output_filename=tfrecord_testing_filepath, is_testing=True)

		metis_db_session.register_tfrecords_volumes(neurodata_client.learner_id, tfrecord_training_filepath,
													tfrecord_validation_filepath, tfrecord_testing_filepath)
		metis_db_session.register_tfrecords_schemata(neurodata_client.learner_id, training_tfrecords_schema,
													 validation_tfrecords_schema, testing_tfrecords_schema)
		metis_db_session.register_tfrecords_examples_num(neurodata_client.learner_id, training_data_size,
														 validation_data_size, testing_data_size)
		metis_db_session.register_tfdatasets_schemata(neurodata_client.learner_id)

	# After generating the tfrecords for training and validation, we need to sketch the distribution of training and
	# validation examples across all federation clients and save the statistics in the result set.
	federation_data_distribution = defaultdict(dict)
	for neurodata_client in neurodata_clients:
		train_task_values = neurodata_client.compute_dataset_stats(task_column="age_at_scan", is_training=True)
		valid_task_values = neurodata_client.compute_dataset_stats(task_column="age_at_scan", is_validation=True)

		federation_data_distribution[neurodata_client.learner_id] = defaultdict(dict)
		federation_data_distribution[neurodata_client.learner_id]['train_stats']['dataset_size'] = \
			len(train_task_values)
		federation_data_distribution[neurodata_client.learner_id]['train_stats']['dataset_values'] = \
			train_task_values

		federation_data_distribution[neurodata_client.learner_id]['validation_stats']['dataset_size'] = \
			len(valid_task_values)
		federation_data_distribution[neurodata_client.learner_id]['validation_stats']['dataset_values'] = \
			valid_task_values

	# Run federation.
	nnmodel = BrainAgeCNNFedModel(distribution_based_training=False)
	federation_rounds = fedexec.FedExecutionOps.federated_between_graph_replication(fed_environment=fed_environment,
																					federation_model_obj=nnmodel,
																					metis_db_session=metis_db_session,
																					session_target_stat_name='mse',
																					batch_level_log=BATCH_LVL_LOGGING)
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
	federation_results = federation_model_execution(fed_environment)

	federation_execution_results = dict()
	federation_environment = yaml.safe_load(open(args.federation_environment_filepath))
	# federation_environment = JSONUtil.lower_keys(federation_environment)
	federation_execution_results["federation_environment"] = federation_environment
	federation_execution_results["federation_results"] = federation_results
	current_timestamp = str(int(TimeUtil.current_milli_time()))
	experiments_log_filename = args.federation_environment_filepath.split("/")[-1].replace('.yaml', '')
	experiments_log_filename += ".semisynchronous_{}.k_{}.ts_{}".format(SEMI_SYNCHRONOUS_EXECUTION,
																		SEMI_SYNCHRONOUS_K_VALUE,
																		current_timestamp)
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
	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	main(args)
	METIS_CATALOG_CLIENT.shutdown()
