import collections

import numpy as np
import tensorflow as tf

from collections import OrderedDict
from utils.logging.metis_logger import MetisLogger as metis_logger


class TFDatasetUtils(object):

	class TFDatasetStructure(object):

		def __init__(self, dataset_init_op, dataset_iterator, dataset_next, dataset_size):
			self.dataset_init_op = dataset_init_op
			self.dataset_iterator = dataset_iterator
			self.dataset_next = dataset_next
			self.dataset_size = dataset_size


	@classmethod
	def _int64_feature(cls, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


	@classmethod
	def _float_feature(cls, value):
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


	@classmethod
	def _bytes_feature(cls, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


	@classmethod
	def _generate_tffeature(cls, dataset_records):
		# Loop over the schema keys.
		record_keys = dataset_records.keys()
		# We split the input arrays in one-to-one examples.
		records_chunks = list()
		for k in record_keys:
			np_array = dataset_records[k]
			records_chunks.append(np.array_split(np_array, np_array.shape[0]))

		# Per example generator yield.
		for chunk in zip(*records_chunks):
			feature = {}
			# Convert every attribute to tf compatible feature.
			for k_idx, k in enumerate(record_keys):
				feature[k] = cls._bytes_feature(tf.compat.as_bytes(chunk[k_idx].flatten().tostring()))
			yield feature


	@classmethod
	def write_ndarrays_to_tfrecords(cls, x_input, y_output, filename):

		assert isinstance(x_input, np.ndarray)
		assert isinstance(y_output, np.ndarray)

		writer = tf.python_io.TFRecordWriter(filename)

		# We split the input arrays in one-to-one examples
		x_chunks = np.array_split(x_input, x_input.shape[0])
		y_chunks = np.array_split(y_output, y_output.shape[0])

		# We zip the input with the output and create serialized examples
		for x_chunk, y_chunk in zip(x_chunks, y_chunks):

			# Since the input is a numpy array we need to serialize it as a string
			example = tf.train.Example(features=tf.train.Features(feature={
				'x_raw': cls._bytes_feature(tf.compat.as_bytes(x_chunk.flatten().tostring())),
				'y_raw': cls._bytes_feature(tf.compat.as_bytes(y_chunk.flatten().tostring()))
			}))

			# Serialize the example to a string
			serialized = example.SerializeToString()

			# Write the serialized object to the file
			writer.write(serialized)

		writer.close()


	# @classmethod
	# def deserialize_single_tfrecord_example(cls, example_proto):
	#
	# 	feature_description = {
	# 		'x_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
	# 		'y_raw': tf.FixedLenFeature(shape=[], dtype=tf.string)
	# 	}
	#
	# 	deserialized_example = tf.io.parse_single_example(serialized=example_proto,
	# 													  features=feature_description)
	#
	# 	x_restored = tf.decode_raw(deserialized_example['x_raw'], tf.float32)
	# 	y_restored = tf.decode_raw(deserialized_example['y_raw'], tf.int32)
	# 	y_restored = tf.reshape(y_restored, shape=())
	#
	# 	return x_restored, y_restored



	@classmethod
	def deserialize_single_tfrecord_example(cls, example_proto: tf.Tensor, example_schema: dict):
		"""
		If the input schema is already ordered then do not change keys order
		and use this sequence to deserialize the records. Else sort the keys
		by name and use the alphabetical sequence to deserialize.
		:param example_proto:
		:param example_schema:
		:return:
		"""
		assert isinstance(example_proto, tf.Tensor)
		assert isinstance(example_schema, dict)

		if not isinstance(example_schema, collections.OrderedDict):
			schema_attributes_positioned = list(sorted(example_schema.keys()))
		else:
			schema_attributes_positioned = list(example_schema.keys())

		feature_description = dict()
		for attr in schema_attributes_positioned:
			feature_description[attr] = tf.FixedLenFeature(shape=[], dtype=tf.string)

		deserialized_example = tf.io.parse_single_example(serialized=example_proto,
														  features=feature_description)
		record = []
		for attr in schema_attributes_positioned:
			attr_restored = tf.decode_raw(deserialized_example[attr], example_schema[attr])
			record.append(attr_restored)

		return record


	@classmethod
	def serialize_to_tfrecords(cls, dataset_records_mappings: dict, output_filename: str):
		"""
		The `dataset_records_mappings` is a dictionary with format:
			{"key1" -> np.ndarray(), "key2" -> np.ndarray(), etc...}
		Using this dict we zip ndarrays rows and we serialize them as tfrecords
		to the output_filename. The schema (attributes) of the serialized tfrecords
		is based on the dictionary keys. The order of the keys in the input dictionary is
		preserved and is used to serialize to tfrecords.
		:param dataset_records_mappings:
		:param output_filename:
		:return:
		"""
		assert isinstance(dataset_records_mappings, dict)
		for val in dataset_records_mappings.values():
			assert isinstance(val, np.ndarray)

		# Attributes tf.data_type is returned in alphabetical order
		tfrecords_schema = collections.OrderedDict({attr: tf.as_dtype(val.dtype.name)
													for attr, val in dataset_records_mappings.items()})

		metis_logger.info("Serializing input data with TF Schema: {} to .tfrecords: {}".format(
			tfrecords_schema, output_filename))

		# Open file writer
		tf_record_writer = tf.python_io.TFRecordWriter(output_filename)
		# Iterate over dataset's features generator
		for feature in cls._generate_tffeature(dataset_records_mappings):
			example = tf.train.Example(features=tf.train.Features(feature=feature))
			# Serialize the example to a string
			serialized = example.SerializeToString()
			# Write the serialized object to the file
			tf_record_writer.write(serialized)
		# Close file writer
		tf_record_writer.close()

		metis_logger.info("Serialization Finished")

		return tfrecords_schema


	@classmethod
	def to_ordered_tfschema(cls, input_schemata):
		# Each schema is expected to be: [[ (<attribute_name_1>, <attribute_tfdatatype_1>), (...), (...)) ]]
		return [OrderedDict(eval(schema)) for schema in input_schemata if schema is not None]


	@classmethod
	def structure_tfdataset(cls, dataset, batch_size, num_examples, shuffle=False):

		if dataset is None:
			dataset = tf.data.Dataset.from_tensor_slices([0])
			dataset = dataset.take(0)

		# TODO Shuffling should be handled externally - Otherwise a preemption error is thrown
		if shuffle:
			shuffle_buffer_size = num_examples
			# shuffle_buffer_size = 250
			dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
		dataset = dataset.batch(batch_size)
		prefetch_buffer_size = batch_size
		dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
		dataset_iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types,
														   output_shapes=dataset.output_shapes)
		get_next_dataset = dataset_iterator.get_next()
		dataset_init_op = dataset_iterator.make_initializer(dataset)
		dataset_ops = cls.TFDatasetStructure(dataset_init_op=dataset_init_op,
											 dataset_iterator=dataset_iterator,
											 dataset_next=get_next_dataset,
											 dataset_size=num_examples)
		return dataset_ops