import collections
import random
import copy
import os

import tensorflow as tf
import nibabel as nib
import pandas as pd
import numpy as np

from utils.tf.tf_ops_dataset import TFDatasetUtils
from metisdb.metisdb_dataset_client import MetisDatasetClient


class NeuroDatasetClient(MetisDatasetClient):

	def __init__(self, learner_id, num_channels=1, row_offset=0, col_offset=0, depth_offset=0,
				 rows=50, cols=50, depth=50):

		super().__init__(learner_id)
		self.num_channels = num_channels
		self.row_offset = row_offset
		self.col_offset = col_offset
		self.depth_offset = depth_offset

		self.rows = rows
		self.cols = cols
		self.depth = depth

		self.__training_data_mappings = None
		self.__validation_data_mappings = None
		self.__test_data_mappings = None


	def client_has_validation_data(self):
		return self.__validation_data_mappings


	def parse_data_mappings_file(self, filepath, data_volume_column, csv_reader_schema,
								 is_training=False, is_validation=False, is_testing=False):
		if not os.path.exists(filepath):
			raise RuntimeError("Error: Provided file: {} does not exist!".format(filepath))
		try:
			mappings_pd = pd.read_csv(filepath, dtype=csv_reader_schema)
		except:
			raise RuntimeError("Error: Provided file: {} is not a valid csv file!".format(filepath))
		if not any([is_training, is_validation, is_testing]):
			raise RuntimeError("Please indicate whether the mapping file is training/validation/testing")

		# Allow for convenient pred (No AgeAtScan).
		if not "age_at_scan" in mappings_pd.columns:
			mappings_pd["age_at_scan"] = -1

		# List the csv file columns.
		csv_columns = list(csv_reader_schema.keys())
		csv_columns.append(data_volume_column)

		# Check whether any required columns are absent.
		required_columns = set(csv_columns)
		existing_columns = set(mappings_pd.columns)
		if not required_columns.issubset(existing_columns):
			raise RuntimeError("Error: Missing columns in table: {}"
				.format(required_columns-existing_columns))

		# DO NOT CHANGE INPUT KEY ORDER:
		# Sort the required columns list based on the attributes
		# position in the input dictionary.
		required_columns = list(sorted(set(required_columns), key=csv_columns.index))
		mappings_pd = mappings_pd[required_columns]

		extracted_mappings = collections.OrderedDict()
		for col in required_columns:
			extracted_mappings[col] = mappings_pd[col].values

		if is_training:
			self.__training_data_mappings = extracted_mappings
		elif is_validation:
			self.__validation_data_mappings = extracted_mappings
		elif is_testing:
			self.__test_data_mappings = extracted_mappings


	def compute_dataset_stats(self, task_column, is_training=False, is_validation=False, is_testing=False):
		if is_training:
			data_mappings = self.__training_data_mappings
		elif is_validation:
			data_mappings = self.__validation_data_mappings
		elif is_testing:
			data_mappings = self.__test_data_mappings

		if data_mappings is None:
			task_values = []
		else:
			serialization_data_mappings = copy.deepcopy(data_mappings)
			task_values = serialization_data_mappings.pop(task_column).tolist()
		return task_values


	def generate_tfrecords(self, data_volume_column, tfrecord_output_filename,
						   is_training=False, is_validation=False, is_testing=False):
		if not any([is_training, is_validation, is_testing]):
			raise RuntimeError("Please indicate whether the generating tfrecord file is training/validation/testing")

		if is_training:
			data_mappings = self.__training_data_mappings
		elif is_validation:
			data_mappings = self.__validation_data_mappings
		elif is_testing:
			data_mappings = self.__test_data_mappings

		if data_mappings is None:
			return None, collections.OrderedDict()
		else:
			serialization_data_mappings = copy.deepcopy(data_mappings)
			examples_num = len(serialization_data_mappings[data_volume_column])

			# serialization_data_mappings[data_volume_column] = \
			# 	serialization_data_mappings[data_volume_column].astype('U')

			# We perform a `pop` operation, because we do not want
			# to serialize the volume attribute of each data file.
			cpaths = serialization_data_mappings.pop(data_volume_column)
			processed_images = []
			for gziped_img in cpaths:
				img = nib.load(gziped_img).get_fdata()
				# # Crop as required.
				# s_r = self.row_offset
				# e_r = s_r + self.rows
				#
				# s_c = self.col_offset
				# e_c = s_c + self.cols
				#
				# s_d = self.depth_offset
				# e_d = s_d + self.depth
				#
				# img = img[s_r:e_r, s_c:e_c, s_d:e_d]
				img = (img - img.mean()) / img.std()
				img = np.float32(img[:, :, :, np.newaxis])
				processed_images.append(img)

			processed_images = np.asarray(processed_images, dtype=np.float32)
			serialization_data_mappings['scan_images'] = processed_images

			tfrecords_schema = TFDatasetUtils.serialize_to_tfrecords(serialization_data_mappings,
																	 tfrecord_output_filename)
			return examples_num, tfrecords_schema


	def load_tfrecords(self, tfrecords_schema, tfrecord_output_filename, is_training):

		# Load dataset from tfrecords.
		dataset = tf.data.TFRecordDataset(tfrecord_output_filename)

		# Parse the record into tensors.
		dataset = dataset.map(map_func=lambda x: TFDatasetUtils
							  .deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecords_schema),
							  num_parallel_calls=3)

		# Transform dataset records for network input.
		dataset = dataset.map(lambda x, y: self.input_record_transformation(x, y, is_training),
							  num_parallel_calls=3)

		# Bundle tensors together as key-value attributes.
		dataset = tf.data.Dataset.zip(({'age': dataset.map(map_func=lambda x, y, z: x),
										'dist': dataset.map(map_func=lambda x, y, z: y),
										'images': dataset.map(map_func=lambda x, y, z: z)}))

		return dataset


	def x_train_input_name(self):
		return "images"


	def y_train_output_name(self):
		return "age"


	def x_eval_input_name(self):
		return "images"


	def y_eval_output_name(self):
		return "age"


	def input_record_transformation(self, x, y, is_training):
		age = x
		dist = self.tf_age_distribution_generation(age)
		processed_img = tf.reshape(y, [self.rows, self.cols, self.depth, self.num_channels])
		# if is_training:
		# 	processed_img = self.tf_neuro_image_augment(z)
		# else:
		# 	processed_img = tf.reshape(z, [self.rows, self.cols, self.depth, self.num_channels])

		return age, dist, processed_img


	def tf_age_distribution_generation(self, age):

		# Hard-coded bins(36) for UKBB
		# with mean at age
		x = tf.range(45.0, 81.0)

		xU = tf.add(x, 0.05)
		xL = tf.subtract(x, 0.05)

		distU = tf.distributions.Normal(loc=age, scale=5.5)
		cdfU = distU.cdf(xU)

		distL = tf.distributions.Normal(loc=age, scale=5.5)
		cdfL = distL.cdf(xL)

		dist = tf.subtract(cdfU, cdfL)
		dist = tf.divide(dist, tf.reduce_sum(dist))

		return dist


	def tf_neuro_image_augment(self, img):
		"""
		Perturb by [0, 2] pixels
		Mirror 50% chance
		:param img:
		:return:
		"""

		img = tf.reshape(img, [self.rows, self.cols, self.depth])

		# Magnitude of shifts in each axis [0, 2]
		ax = random.choice([0, 1, 2, 42])
		mg = random.choice([-2, -1, 1, 2])

		if ax == 0 and mg > 0:
			img = tf.pad(img, ((mg, 0), (0, 0), (0, 0)), mode="constant")[:-mg, :, :]
		elif ax == 0 and mg < 0:
			img = tf.pad(img, ((0, -mg), (0, 0), (0, 0)), mode="constant")[-mg:, :, :]

		elif ax == 1 and mg > 0:
			img = tf.pad(img, ((0, 0), (mg, 0), (0, 0)), mode="constant")[:, :-mg, :]
		elif ax == 1 and mg < 0:
			img = tf.pad(img, ((0, 0), (0, -mg), (0, 0)), mode="constant")[:, -mg:, :]

		elif ax == 2 and mg > 0:
			img = tf.pad(img, ((0, 0), (0, 0), (mg, 0)), mode="constant")[:, :, :-mg]
		elif ax == 2 and mg < 0:
			img = tf.pad(img, ((0, 0), (0, 0), (0, -mg)), mode="constant")[:, :, -mg:]

		img = tf.reshape(img, [self.rows, self.cols, self.depth, self.num_channels])

		# # Random mirroring
		# if random.choice([True, False]):
		# 	img = tf.reverse(img, axis=0)

		return img
