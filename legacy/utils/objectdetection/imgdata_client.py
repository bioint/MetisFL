import collections
import datetime
import os
import sys

import metisdb.postgres_client as pg_client
import numpy as np
import tensorflow as tf

from utils.generic.data_partitioning_ops import DataPartitioningUtil
from metisdb.metisdb_dataset_client import MetisDatasetClient
from utils.logging.metis_logger import MetisLogger as metis_logger
from utils.tf.tf_ops_dataset import TFDatasetUtils

ImgDatasetFields = collections.namedtuple(typename='ImgDatasetFields', field_names=['input', 'output', 'size'])
ImgDatasetsCollections = collections.namedtuple(typename='ImgDatasetsCollections', field_names=['train', 'test'])

class ImgDatasetLoader(object):

	__ImgDatasets = None
	__ImgDatasetsCache = False

	def __init__(self, mnist_loader=False, fmnist_loader=False, extended_mnist_loader=False, cifar10_loader=False,
				 cifar100_loader=False, train_examples=100000, test_examples=100000, *args, **kwargs):

		if mnist_loader is False and fmnist_loader is False and extended_mnist_loader is False \
				and cifar10_loader is False and cifar100_loader is False:
			raise RuntimeError('At least one of the session parameters must be set to True')

		if mnist_loader:
			self.num_classes = 10
			self.is_eval_output_single_scalar = True
			self.negative_classes_indices = []
			self.__DBName = 'MnistDB'
			self.__training_table = 'mnistdata_training'
			self.__testing_table = 'mnistdata_testing'
		elif fmnist_loader:
			self.num_classes = 10
			self.is_eval_output_single_scalar = True
			self.negative_classes_indices = []
			self.__DBName = 'FMnistDB'
			self.__training_table = 'fmnistdata_training'
			self.__testing_table = 'fmnistdata_testing'
		elif extended_mnist_loader:
			self.is_eval_output_single_scalar = True
			self.negative_classes_indices = []
			self.__DBName = 'ExtendedMnistDB'
			if "extended_mnist_byclass" in kwargs and kwargs["extended_mnist_byclass"] is True:
				self.num_classes = 62
				self.__training_table = 'extended_mnistdata_byclass_training'
				self.__testing_table = 'extended_mnistdata_byclass_testing'
			elif "extended_mnist_bymerge" in kwargs and kwargs["extended_mnist_bymerge"] is True:
				self.num_classes = 47
				self.__training_table = 'extended_mnistdata_bymerge_training'
				self.__testing_table = 'extended_mnistdata_bymerge_testing'
			elif "extended_mnist_balanced" in kwargs and kwargs["extended_mnist_balanced"] is True:
				self.num_classes = 47
				self.__training_table = 'extended_mnistdata_balanced_training'
				self.__testing_table = 'extended_mnistdata_balanced_testing'
			elif "extended_mnist_letters" in kwargs and kwargs["extended_mnist_letters"] is True:
				self.num_classes = 26
				self.__training_table = 'extended_mnistdata_letters_training'
				self.__testing_table = 'extended_mnistdata_letters_testing'
			elif "extended_mnist_digits" in kwargs and kwargs["extended_mnist_digits"] is True:
				self.num_classes = 10
				self.__training_table = 'extended_mnistdata_digits_training'
				self.__testing_table = 'extended_mnistdata_digits_testing'
			else:
				raise RuntimeError('At least one of the extended mnist data type parameters must be set to True.')
		elif cifar10_loader:
			self.num_classes = 10
			self.is_eval_output_single_scalar = True
			self.negative_classes_indices = []
			self.__DBName = 'Cifar10DB'
			self.__training_table = 'cifar10data_training'
			self.__testing_table = 'cifar10data_testing'
		elif cifar100_loader:
			self.num_classes = 100
			self.is_eval_output_single_scalar = True
			self.negative_classes_indices = []
			self.__DBName = 'Cifar100DB'
			self.__training_table = 'cifar100data_training'
			self.__testing_table = 'cifar100data_testing'

		self.train_examples_num = train_examples
		self.test_examples_num = test_examples
		self.partitions_num = None
		self.partition_policy = None
		self.is_partitioned = False
		self.partitioned_training_data = dict()


	def __fetch_from_metisdb(self, tblname=None, examples=100):
		"""
		A helper function to query the Metis Database. This function invokes the underlying SQL engine
		Args:
			tblname: The Name of the table to fetch the data from
			examples: Number of examples to retrive form the table

		Returns:
			A list of index-wise-associated images and labels
		"""

		psqlpool = pg_client.get_postgres_connection_pool(dbname=self.__DBName)
		active_conn = psqlpool.getconn()
		input_data, labels = pg_client.select_metisdb_data(psqlconn=active_conn, tblname=tblname, limit=examples)
		psqlpool.closeall()
		return input_data, labels


	def __load_image_datasets(self):

		# Load Train,Test Data from DB
		train_input, train_labels = self.__fetch_from_metisdb(tblname=self.__training_table, examples=self.train_examples_num)
		test_input, test_labels = self.__fetch_from_metisdb(tblname=self.__testing_table, examples=self.test_examples_num)

		# Train data
		train_input = np.asarray(train_input[:self.train_examples_num], dtype=np.float32)
		train_labels = np.asarray(train_labels[:self.train_examples_num], dtype=np.int32)
		train_size = train_labels.size

		# Test Data
		test_input = np.asarray(test_input, dtype=np.float32)
		test_labels = np.asarray(test_labels, dtype=np.int32)
		test_size = test_labels.size

		# Data has been retrieved from the database, initialize the cache
		self.__ImgDatasetsCache = True
		self.__ImgDatasets = ImgDatasetsCollections(
			train=ImgDatasetFields(input=train_input, output=train_labels, size=train_size),
			test=ImgDatasetFields(input=test_input, output=test_labels, size=test_size))


	def load_image_datasets(self):
		"""
		A dataset loader helper function.

		Returns:

		"""
		self.__load_image_datasets()
		return self.__ImgDatasets


	def get_all_training_data(self):
		assert self.__ImgDatasetsCache is True, "Need to initialize data cache"
		return self.__ImgDatasets.train


	def get_all_test_data(self):
		assert self.__ImgDatasetsCache is True, "Need to initialize data cache"
		return self.__ImgDatasets.test


	def partition_training_data(self, partitions_num, balanced_random_partitioning=False,
								balanced_class_partitioning=False, unbalanced_class_partitioning=False,
								skewness_factor=None, strictly_unbalanced=None,
								classes_per_partition=None, MLSYS_REBUTTAL_REVIEWER2=False):
		"""
		A helper function to partition the data based on classes.
		Args:
			partitions_num:
			balanced_random_partitioning:
			balanced_class_partitioning:
			unbalanced_class_partitioning:
			skewness_factor:
			strictly_unbalanced:
			classes_per_partition:

		Returns:

		"""
		# If datasets have not been retrieved yet, then fetch the projectmetis data and store them in the cache
		if not self.__ImgDatasetsCache:
			raise RuntimeError("First need to initialize data cache, invoke %s()" % self.load_image_datasets.__name__)

		if not isinstance(partitions_num, int):
			raise TypeError("`partitions_num` must be an integer")
		else:
			self.partitions_num = partitions_num

		# Erase and create new data partitions
		self.partitioned_training_data = dict()
		self.partition_policy = "IID" if classes_per_partition == self.num_classes else "Non-IID"


		# TODO HACK!!!
		if MLSYS_REBUTTAL_REVIEWER2:
			self.partition_policy = "IID"
			self.partitioned_training_data = DataPartitioningUtil.balanced_class_partitioning(
				input_data=self.__ImgDatasets.train.input,
				output_classes=self.__ImgDatasets.train.output,
				partitions=2,
				classes_per_partition=5)
			zero_to_four = self.partitioned_training_data[0]
			zero_to_four_partitioned_training_data = DataPartitioningUtil.balanced_class_partitioning(
				input_data=zero_to_four.input,
				output_classes=zero_to_four.output,
				partitions=5,
				classes_per_partition=5)
			five_to_nine = self.partitioned_training_data[1]
			five_to_nine_partitioned_training_data = DataPartitioningUtil.balanced_class_partitioning(
				input_data=five_to_nine.input,
				output_classes=five_to_nine.output,
				partitions=5,
				classes_per_partition=5)
			self.partitioned_training_data = dict()
			j = 0
			for i in range(0, 10, 2):
				self.partitioned_training_data[i] = five_to_nine_partitioned_training_data[j]
				self.partitioned_training_data[i+1] = zero_to_four_partitioned_training_data[j]
				j += 1

		elif balanced_random_partitioning:
			self.partition_policy = "IID"
			self.partitioned_training_data = DataPartitioningUtil.balanced_random_partitioning(
				input_data=self.__ImgDatasets.train.input,
				output_classes=self.__ImgDatasets.train.output,
				partitions=partitions_num)

		elif balanced_class_partitioning:
				if classes_per_partition is None:
					raise RuntimeError("`classes_per_partition` must be defined when class partitioning is invoked")
				self.partitioned_training_data = DataPartitioningUtil.balanced_class_partitioning(
					input_data=self.__ImgDatasets.train.input,
					output_classes=self.__ImgDatasets.train.output,
					partitions=partitions_num,
					classes_per_partition=classes_per_partition)

		elif unbalanced_class_partitioning is True and skewness_factor is not None:

				if skewness_factor is None:
					raise RuntimeError("You need to specify the skewness factor of the distribution.")
				if strictly_unbalanced is None:
					raise RuntimeError("You need to specify if you want to create a strictly unbalanced distribution.")

				if strictly_unbalanced is False:
					self.partitioned_training_data = DataPartitioningUtil.strictly_noniid_unbalanced_data_partitioning(
						input_data=self.__ImgDatasets.train.input,
						output_classes=self.__ImgDatasets.train.output,
						partitions=partitions_num,
						classes_per_partition=classes_per_partition,
						skewness_factor=skewness_factor)
				else:
					self.partitioned_training_data = DataPartitioningUtil.strictly_unbalanced_noniid_data_partitioning(
						input_data=self.__ImgDatasets.train.input,
						output_classes=self.__ImgDatasets.train.output,
						partitions=partitions_num,
						classes_per_partition=classes_per_partition,
						skewness_factor=skewness_factor)

		else:
			raise RuntimeError("You need to specify at least one data partitioning scheme")

		partition_ids_by_descending_partition_size = sorted(
			self.partitioned_training_data,
			key=lambda partition_id: self.partitioned_training_data[partition_id].partition_size, reverse=True)
		new_sorted_session_partitioned_data = dict()
		for new_pidx, partition_id in enumerate(partition_ids_by_descending_partition_size):
			new_sorted_session_partitioned_data[new_pidx] = self.partitioned_training_data[partition_id]

		self.partitioned_training_data = new_sorted_session_partitioned_data
		self.is_partitioned = True

		for pidx, pidx_values in self.partitioned_training_data.items():
			metis_logger.info("Partition ID:{}, Data Distribution: {}".format(pidx, pidx_values))

		return self.partitioned_training_data


	def retrieve_all_partitions_training_dataset_stats(self, to_json_representation=False, partitions_labels=list()):
		all_partitions_dataset_stats = dict()
		if self.is_partitioned:
			if len(partitions_labels) == 0:
				partitions_labels = sorted(self.partitioned_training_data.keys())
			for pidx, pid in enumerate(self.partitioned_training_data.keys()):
				if to_json_representation:
					all_partitions_dataset_stats[partitions_labels[pidx]] \
						= self.partitioned_training_data[pid].toJSON_representation()
				else:
					all_partitions_dataset_stats[partitions_labels[pidx]] = self.partitioned_training_data[pid]
		return all_partitions_dataset_stats


class ImgDatasetClient(MetisDatasetClient):

	def __init__(self, image_dataset_loader, learner_id, learner_partition_idx=None, validation_percentage=0.0,
				 distort_training_images=True, mnist_client=False, fmnist_client=False, extended_mnist_client=False,
				 cifar10_client=False, cifar100_client=False, corrupt_labels_uniform=False, corrupt_images_opposite=False,
				 *args, **kwargs):

		assert isinstance(image_dataset_loader, ImgDatasetLoader), "Need to initialize the ImgDatasetLoader"
		if mnist_client is False and fmnist_client is False and extended_mnist_client is False \
				and cifar10_client is False and cifar100_client is False:
			raise RuntimeError('At least one of the client type parameters must be set to True')

		super().__init__(learner_id)
		self.image_dataset_loader = image_dataset_loader
		self.learner_partition_idx = learner_partition_idx

		self.__isMnistSession = mnist_client
		self.__isFMnistSession = fmnist_client
		self.__isExtMnistSession = extended_mnist_client
		self.__isCifar10Session = cifar10_client
		self.__isCifar100Session = cifar100_client
		if self.__isMnistSession:
			self.__image_channels = 1
			self.__original_image_height = 28
			self.__original_image_width = 28
			self.num_classes = 10
		if self.__isFMnistSession:
			self.__image_channels = 1
			self.__original_image_height = 28
			self.__original_image_width = 28
			self.num_classes = 10
		if self.__isExtMnistSession:
			self.__image_channels = 1
			self.__original_image_height = 28
			self.__original_image_width = 28
		if self.__isCifar10Session:
			self.__image_channels = 3
			self.__original_image_height = 32
			self.__original_image_width = 32
			self.__distorted_image_height = 24
			self.__distorted_image_width = 24
			self.num_classes = 10
		if self.__isCifar100Session:
			self.__image_channels = 3
			self.__original_image_height = 32
			self.__original_image_width = 32
			self.__distorted_image_height = 24
			self.__distorted_image_width = 24
			self.num_classes = 100

		self.tf_func_num_parallel_calls = 20
		self.distort_training_images = distort_training_images
		self.validation_percentage = validation_percentage

		# Careful the following refers to the PartitionDataset class!
		self.learner_training_data = self.image_dataset_loader.partitioned_training_data[self.learner_partition_idx]
		self.learner_validation_data = None
		# Careful the following refers to the ImgDatasetFields collection!
		self.learner_testing_data = self.image_dataset_loader.get_all_test_data()

		# TODO Add support for corrupted local data
		if corrupt_labels_uniform:
			true_labels = self.learner_training_data.output
			rng = np.random.default_rng()
			min_label, max_label = np.nanmin(true_labels), np.nanmax(true_labels)
			corrupted_labels = rng.integers(low=min_label, high=max_label, size=true_labels.size, endpoint=True)
			self.learner_training_data.output = corrupted_labels
			metis_logger.info("Uniform labels corruption for: {}".format(str(self.learner_id)))
		if corrupt_images_opposite:
			true_images = self.learner_training_data.input
			corrupted_images = 1. - true_images
			self.learner_training_data.input = corrupted_images
			metis_logger.info("Opposite images corruption for: {}".format(str(self.learner_id)))

		if validation_percentage != 0.0:
			partition_training_validation_datasets = DataPartitioningUtil \
				.stratified_training_holdout_dataset(partition_data=self.learner_training_data,
													 holdout_proportion=validation_percentage)
			# Careful the following refers to the PartitionDataset class!
			self.learner_training_data = partition_training_validation_datasets[0]
			self.learner_validation_data = partition_training_validation_datasets[1]


	def __cifar_images_preprocessing(self, images, is_training_dataset=True):
		"""
		An image processing function for the Cifar10 tf.data.Dataset.
		Args:
			images:
			is_training_dataset:

		Returns:

		"""
		if not isinstance(images, tf.data.Dataset):
			raise RuntimeError("The provided %s must be of type %s" % (type(images), tf.data.Dataset))

		height = self.__distorted_image_height if self.distort_training_images else self.__original_image_height
		width = self.__distorted_image_width if self.distort_training_images else self.__original_image_width
		depth = self.__image_channels

		# First need to reshape the images to their original dimensions
		images = images.map(map_func=lambda img: tf.reshape(
							img, [self.__original_image_height, self.__original_image_width, self.__image_channels]),
							num_parallel_calls=self.tf_func_num_parallel_calls)

		# Image processing for training the network.
		# Note the many random distortions applied to the image.
		if is_training_dataset and self.distort_training_images:

			distorted_images = images.map(map_func=lambda img: tf.random_crop(img, [height, width, depth]),
										  num_parallel_calls=self.tf_func_num_parallel_calls)
			distorted_images = distorted_images.map(map_func=lambda img: tf.image.random_flip_left_right(img),
													num_parallel_calls=self.tf_func_num_parallel_calls)

			# Because these operations are not commutative, consider randomizing
			# the order their operation.
			# NOTE: since per_image_standardization zeros the mean and makes
			# the stddev unit, this likely has no effect see tensorflow#1458.
			distorted_images = distorted_images.map(map_func=lambda img: tf.image.random_brightness(img, max_delta=63),
													num_parallel_calls=self.tf_func_num_parallel_calls)
			images = distorted_images.map(map_func=lambda img: tf.image.random_contrast(img, lower=0.2, upper=1.8),
										  num_parallel_calls=self.tf_func_num_parallel_calls)

		# Image processing for evaluation.
		if not is_training_dataset:
			# Crop the central [height, width] of the image.
			images = images.map(map_func=lambda img: tf.image.resize_image_with_crop_or_pad(img, height, width),
								num_parallel_calls=self.tf_func_num_parallel_calls)

		# Subtract off the mean and divide by the variance of the pixels.
		images = images.map(map_func=lambda img: tf.image.per_image_standardization(img),
							num_parallel_calls=self.tf_func_num_parallel_calls)

		return images


	def __mnist_images_preprocessing(self, images):
		"""
		An image processing function for the Mnist tf.data.Dataset.
		Args:
			images:

		Returns:

		"""
		if not isinstance(images, tf.data.Dataset):
			raise RuntimeError("The provided %s must be of type %s" % (type(images), tf.data.Dataset))

		images = images.map(map_func=lambda img: tf.multiply(img, 1.0 / 255.0),
							num_parallel_calls=self.tf_func_num_parallel_calls)
		images = images.map(map_func=lambda img: tf.reshape(
							img, [self.__original_image_height, self.__original_image_width, self.__image_channels]),
							num_parallel_calls=self.tf_func_num_parallel_calls)

		return images


	def generate_tfrecords(self, tfrecord_output_filename, is_training=False, is_validation=False, is_testing=False):

		examples_num = None
		tfrecords_schema = collections.OrderedDict()
		if is_training:
			examples_num = self.learner_training_data.training_data_size
			serialization_mappings = {'images': self.learner_training_data.input,
									  'labels': self.learner_training_data.output}
			tfrecords_schema = TFDatasetUtils.serialize_to_tfrecords(serialization_mappings, tfrecord_output_filename)

		if is_validation and self.learner_validation_data is not None:
			examples_num = self.learner_validation_data.training_data_size
			serialization_mappings = {'images': self.learner_validation_data.input,
									  'labels': self.learner_validation_data.output}
			tfrecords_schema = TFDatasetUtils.serialize_to_tfrecords(serialization_mappings, tfrecord_output_filename)

		if is_testing:
			examples_num = self.learner_testing_data.size
			serialization_mappings = {'images': self.learner_testing_data.input,
									  'labels': self.learner_testing_data.output}
			tfrecords_schema = TFDatasetUtils.serialize_to_tfrecords(serialization_mappings, tfrecord_output_filename)

		return examples_num, tfrecords_schema


	def load_tfrecords(self, tfrecords_schema, tfrecord_output_filename, is_training):

		# Load dataset from tfrecords.
		dataset = tf.data.TFRecordDataset(tfrecord_output_filename)

		# Parse the record into tensors.
		dataset = dataset.map(map_func=lambda x: TFDatasetUtils
							  .deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecords_schema),
							  num_parallel_calls=3)
		dataset_x = dataset.map(map_func=lambda x, y: x)
		dataset_y = dataset.map(map_func=lambda x, y: tf.squeeze(y))

		if self.__isCifar10Session or self.__isCifar100Session:
			# Preprocess Cifar10(0) Input
			dataset_x = self.__cifar_images_preprocessing(images=dataset_x,
														  is_training_dataset=is_training)

		elif self.__isMnistSession or self.__isFMnistSession or self.__isExtMnistSession:
			# Preprocess (F/Extended) MNIST Input
			dataset_x = self.__mnist_images_preprocessing(images=dataset_x)

		# Bundle tensors together as key-value attributes.
		dataset = tf.data.Dataset.zip(({'images': dataset_x, 'labels': dataset_y}))

		return dataset


	def x_train_input_name(self):
		return "images"


	def y_train_output_name(self):
		return "labels"


	def x_eval_input_name(self):
		return "images"


	def y_eval_output_name(self):
		return "labels"
