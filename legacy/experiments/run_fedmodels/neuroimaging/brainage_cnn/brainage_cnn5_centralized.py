import argparse
import collections
import datetime
import numpy as np
import os
import random
import tensorflow as tf

from utils.brainage.data_gen import Scan_Gen
from utils.brainage.neurodata_client import NeuroDatasetClient
from experiments.tf_fedmodels.cnn.cnn5_brain_age import BrainAgeCNNFedModel
from utils.evaluation.model_evaluation_metrics import Regression
from utils.tf.tf_ops_dataset import TFDatasetUtils
from metisdb.metisdb_catalog import MetisCatalog
from metisdb.metisdb_session import MetisDBSession
from metisdb.sqlalchemy_client import SQLAlchemyClient

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ["GRPC_VERBOSITY"] = 'ERROR'

BATCH_DEFAULT = 1

np.random.seed(seed=1990)
random.seed(a=1990)
tf.set_random_seed(seed=1990)


# Train the model
def train(args):

	data_identifier = 'subject_id'
	data_task_column = 'age_at_scan'
	data_volume_column = '9dof_2mm_vol'
	LEARNER_ID = "centralized"
	TF_RECORD_TRAIN_FILEPATH = "/tmp/metis_project/execution_logs/neuro.train.tfrecord"
	TF_RECORD_VALIDATION_FILEPATH = "/tmp/metis_project/execution_logs/neuro.validation.tfrecord"
	sqlalchemy_client = SQLAlchemyClient(postgres_instance=True)
	METIS_CATALOG_CLIENT = MetisCatalog(sqlalchemy_client)

	neurodata_client = NeuroDatasetClient(learner_id=LEARNER_ID,
										  depth_offset=0, rows=91, cols=109, depth=91)
	neurodata_client.parse_data_mappings_file(
		filepath=args.train_data_path,
		data_volume_column=data_volume_column,
		csv_reader_schema={data_identifier: np.int32, data_task_column: np.float32, data_volume_column: np.str},
		is_training=True)
	neurodata_client.parse_data_mappings_file(
		filepath=args.valid_data_path,
		data_volume_column=data_volume_column,
		csv_reader_schema={data_identifier: np.int32, data_task_column: np.float32, data_volume_column: np.str},
		is_validation=True)

	print("Starting training tfrecords generation.")
	training_data_size, training_tfrecords_schema = neurodata_client.generate_tfrecords(
		data_volume_column=data_volume_column, tfrecord_output_filename=TF_RECORD_TRAIN_FILEPATH, is_training=True)
	print(training_tfrecords_schema)

	print("Starting validation tfrecords generation.")
	validation_data_size, validation_tfrecords_schema = neurodata_client.generate_tfrecords(
		data_volume_column=data_volume_column, tfrecord_output_filename=TF_RECORD_VALIDATION_FILEPATH, is_validation=True)
	print(validation_tfrecords_schema)

	metis_db_session = MetisDBSession(METIS_CATALOG_CLIENT, [neurodata_client], is_regression=True)
	metis_db_session.register_tfrecords_volumes(LEARNER_ID, TF_RECORD_TRAIN_FILEPATH, TF_RECORD_VALIDATION_FILEPATH)
	metis_db_session.register_tfrecords_schemata(LEARNER_ID, training_tfrecords_schema, validation_tfrecords_schema)
	metis_db_session.register_tfrecords_examples_num(LEARNER_ID, training_data_size, validation_data_size)
	metis_db_session.register_tfdatasets_schemata(LEARNER_ID)


	with tf.device("/gpu:{}".format(args.gpu_offset)):

		# Define global step
		global_step = tf.train.get_or_create_global_step()

		# deep_cnn = cnn_model(placeholder_network_input, is_training)
		nnmodel = BrainAgeCNNFedModel(distribution_based_training=False)
		model_input_tensors = nnmodel.input_tensors_datatype()
		model_output_tensors = nnmodel.output_tensors_datatype()
		model_architecture = nnmodel.model_architecture(input_tensors=model_input_tensors,
														output_tensors=model_output_tensors,
														global_step=global_step)
		train_step = model_architecture.train_step
		loss_tensor = model_architecture.loss
		predictions_tensor = model_architecture.predictions
		model_trainable_variables = model_architecture.model_federated_variables

		# Dataset case 1
		# Create generator object
		# generator = Scan_Gen(data_config=["9dof_2mm_vol"], rows=91, cols=109, depth=91, pthreads=10)
		# training_dataset, ntrain = generator.create_dataset_wschema(
		# 	args.train_data_path, is_training=True, req_col=["subject_id", "age_at_scan"])
		# validation_dataset, nvalid = generator.create_dataset_wschema(
		# 	args.valid_data_path, is_training=False, req_col=["subject_id", "age_at_scan"])

		# Dataset case 2
		# training_dataset = neurodata_client.load_tfrecords(training_tfrecords_schema, TF_RECORD_TRAIN_FILEPATH, True)
		# validation_dataset = neurodata_client.load_tfrecords(validation_tfrecords_schema, TF_RECORD_VALIDATION_FILEPATH, False)

		# train_data_ops = TFDatasetUtils.structure_tfdataset(
		# 	training_dataset, batch_size=BATCH_DEFAULT, num_examples=training_data_size)
		# validation_data_ops = TFDatasetUtils.structure_tfdataset(
		# 	validation_dataset, batch_size=BATCH_DEFAULT, num_examples=validation_data_size)

		# Dataset case 3
		train_data_ops, validation_data_ops, test_data_ops = metis_db_session.import_host_data(
			learner_id=LEARNER_ID, batch_size=BATCH_DEFAULT, import_train=True, import_validation=True,
			import_test=False)

		next_train_dataset = train_data_ops.dataset_next
		train_dataset_init_op = train_data_ops.dataset_init_op

		next_validation_dataset = validation_data_ops.dataset_next
		validation_dataset_init_op = validation_data_ops.dataset_init_op

		def accumulators(ground_truth, prediction):
			ground_truth = tf.squeeze(ground_truth)
			# Necessary condition when calling squeeze operation, because in the presence of a single element, it
			# will return a single element. E.g. tf.squeeze([[[68.31521]]]) -> array(68.31521) NOT array([68.31521]).
			ground_truth = tf.reshape(ground_truth, shape=[tf.size(ground_truth), ])
			prediction = tf.squeeze(prediction)
			prediction = tf.reshape(prediction, shape=[tf.size(prediction), ])

			with tf.variable_scope('model_prediction_terms'):
				# Defining a collection for the accumulators will raise a Variable uninitialized error! So collections is empty!
				ground_truths_var = tf.Variable([np.inf], trainable=False, validate_shape=False, dtype=tf.float32,
												name="GroundTruthAccumulator")
				predictions_var = tf.Variable([np.inf], trainable=False, validate_shape=False, dtype=tf.float32,
											  name="PredictionsAccumulator")
				prediction_terms_reset_op = [ground_truths_var.initializer, predictions_var.initializer]

			# Conditional Assignment. If it is the first time we assign a value to the variable then we need to assign
			# the batch as it is. In essence, we want to remove the np.infinity value from the variable, thus we check
			# whether its first element is equal to np.inf.
			ground_truths_accumulator_update_op = tf.cond(tf.equal(ground_truths_var[0], np.inf),
														  true_fn=lambda: tf.assign(ground_truths_var, ground_truth,
																					validate_shape=False),
														  false_fn=lambda: tf.assign(ground_truths_var, tf.concat(
															  [ground_truths_var, ground_truth], axis=0),
																					 validate_shape=False))
			predictions_accumulator_update_op = tf.cond(tf.equal(predictions_var[0], np.inf),
														true_fn=lambda: tf.assign(predictions_var, prediction,
																				  validate_shape=False),
														false_fn=lambda: tf.assign(predictions_var, tf.concat(
															[predictions_var, prediction], axis=0),
																				   validate_shape=False))
			# Make sure we do not feed/concat empty values
			ground_truths_accumulator_update_op = tf.cond(
				tf.logical_and(
					tf.not_equal(tf.size(ground_truth), 0), tf.not_equal(tf.size(prediction), 0)),
				true_fn=lambda: ground_truths_accumulator_update_op,
				false_fn=lambda: ground_truths_var)

			predictions_accumulator_update_op = tf.cond(
				tf.logical_and(
					tf.not_equal(tf.size(ground_truth), 0), tf.not_equal(tf.size(prediction), 0)),
				true_fn=lambda: predictions_accumulator_update_op,
				false_fn=lambda: predictions_var)

			update_op = [ground_truths_accumulator_update_op, predictions_accumulator_update_op]

			return update_op, prediction_terms_reset_op, ground_truths_var, predictions_var

	update_op, reset_op, ground_truth_list, prediction_list = accumulators(ground_truth=model_output_tensors['age'],
																		   prediction=predictions_tensor.get_tensor())

	init = tf.global_variables_initializer()
	gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=1)
	config = tf.ConfigProto(gpu_options=gpu_options,
							inter_op_parallelism_threads=10,
							intra_op_parallelism_threads=10,
							allow_soft_placement=True)

	with tf.Session(config=config) as sess:

		# Initialize graph variables and operations
		sess.run(init)
		start_epoch = 0

		# Train for num_epochs
		for epoch in range(start_epoch, start_epoch + args.epochs):

			sess.run(train_dataset_init_op)
			batch_idx = 0

			while True:
				try:

			# epoch_def = ntrain // BATCH_DEFAULT + 1
			# for batch in range(epoch_def):

					train_batch = sess.run(next_train_dataset)
					train_extra_feeds = dict()
					train_extra_feeds.update(train_step.get_feed_dictionary())
					for placeholder_name, placeholder_def in model_input_tensors.items():
						train_extra_feeds[placeholder_def] = train_batch[placeholder_name]
					for placeholder_name, placeholder_def in model_output_tensors.items():
						train_extra_feeds[placeholder_def] = train_batch[placeholder_name]

					train_step.run_tf_operation(session=sess, extra_feeds=train_extra_feeds)
					batch_idx += 1

					# if int(train_batch['subject_id'][0]) == 838:
						# print(train_batch['age'].flatten())
						# print(train_batch['images'].flatten()[:100])

					# if batch_idx == (ntrain//BATCH_DEFAULT +1):
					# 	break
					# Use the inference placeholders values when performing an evaluation. Update values
					# of current training step placeholders with their corresponding inference values.
					# train_extra_feeds.update(predictions_tensor.get_feed_dictionary())
					# sess.run(update_op, feed_dict=train_extra_feeds)

				except tf.errors.OutOfRangeError:
					break

			# ground_truth_ages = sess.run(ground_truth_list)
			# predicted_ages = sess.run(prediction_list)
			# regression_metrics = Regression(is_train=True)\
			# 	.retrieve_regression_metrics(ground_truth_ages, predicted_ages)
			# print("{}, Training Epoch: {}, Results: {}".format(datetime.datetime.now(), epoch+1, regression_metrics))
			# sess.run(reset_op)

			batch_idx = 0
			sess.run(validation_dataset_init_op)
			while True:
				try:

			# Init valid iterator
			# sess.run(validation_iterator.initializer)
			# epoch_def_valid = nvalid // BATCH_DEFAULT + 1
			# for batch in range(epoch_def_valid):

					validation_extra_feeds = dict()
					validation_batch = sess.run(next_validation_dataset)
					for placeholder_name, placeholder_def in model_input_tensors.items():
						validation_extra_feeds[placeholder_def] = validation_batch[placeholder_name]
					for placeholder_name, placeholder_def in model_output_tensors.items():
						validation_extra_feeds[placeholder_def] = validation_batch[placeholder_name]

					# Use the inference placeholders values when performing an evaluation.
					validation_extra_feeds.update(predictions_tensor.get_feed_dictionary())
					sess.run(update_op, feed_dict=validation_extra_feeds)
					batch_idx += 1

				except tf.errors.OutOfRangeError:
					break

			ground_truth_ages = sess.run(ground_truth_list)
			predicted_ages = sess.run(prediction_list)
			regression_metrics = Regression(is_validation=True)\
				.retrieve_regression_metrics(ground_truth_ages, predicted_ages)
			print("{}, Validation Epoch: {}, Results: {}".format(datetime.datetime.now(), epoch+1, regression_metrics))
			sess.run(reset_op)


def main(args):

	# Path to saves
	save_path = os.path.join(args.exp_path, "saves")

	# Create exp directory if doesn't exist
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Path to tboard
	if args.tboard:
		tboard_path = os.path.join(args.exp_path, "tboard")

		# Create tboard directory if doesn't exist
		if not os.path.exists(tboard_path):
			os.makedirs(tboard_path)
		else:  # Don't want to mess things up!
			print("tboard exists! Please shift elsewhere!")
			exit(1)

	train(args)


def get_args():
	parser = argparse.ArgumentParser()

	# Required params
	parser.add_argument("--train_data_path", help="path to training data")
	parser.add_argument("--valid_data_path", help="path to validation data")
	parser.add_argument("--exp_path", help="path to exp folder")

	# Training "logistical constraints"
	parser.add_argument(
		"--tboard", action="store_true", default=False, help="make tensorboard")

	# Machine params
	parser.add_argument(
		"--gpu_offset", type=int, default=0,
		help="idx of gpu to use in multi-gpu cluster")
	parser.add_argument(
		"--pthreads", type=int, default=10, help="num threads for generator")

	# Training params
	parser.add_argument(
		"--epochs", type=int, default=100, help="num of epochs to train")


	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	main(args)
