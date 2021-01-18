from metisdb.metisdb_catalog import MetisCatalog
from metisdb.metisdb_dataset_client import MetisDatasetClient
from utils.tf.tf_ops_dataset import TFDatasetUtils


class MetisDBSession(object):


	def __init__(self, metis_catalog_client, metisdb_dataset_clients, is_classification=False, is_regression=False,
				 num_classes=None, negative_classes_indices=None, is_eval_output_scalar=None):

		assert isinstance(metis_catalog_client, MetisCatalog)

		assert isinstance(metisdb_dataset_clients, list) and \
			   all([isinstance(client, MetisDatasetClient) for client in metisdb_dataset_clients])

		assert (is_classification is True or is_regression is True), \
			"Need to specify whether this is a classification or regression evaluation task."

		self.__metis_catalog_client = metis_catalog_client
		self.__metisdb_dataset_clients = metisdb_dataset_clients
		self.is_classification = is_classification
		self.is_regression = is_regression

		# Only for classification tasks
		self.num_classes = num_classes
		self.negative_classes_indices = negative_classes_indices
		self.is_eval_output_scalar = is_eval_output_scalar


	def __get_learner_dataset_client(self, learner_id):
		learner_dataset_client = [client for client in self.__metisdb_dataset_clients
								  if client.learner_id == learner_id]
		if len(learner_dataset_client) >= 1:
			learner_dataset_client = learner_dataset_client[0]
		else:
			raise RuntimeError("The provided learner identifier: {} does not exist/non-valid".format(learner_id))
		return learner_dataset_client


	def register_tfrecords_volumes(self, learner_id, train_volume=None, validation_volume=None, test_volume=None):
		return self.__metis_catalog_client.register_tfrecords_volumes_wcatalog(
			learner_id, train_volume, validation_volume, test_volume)


	def register_tfrecords_examples_num(self, learner_id, train_size=0, validation_size=0, test_size=0):
		return self.__metis_catalog_client.register_tfrecords_sizes_wcatalog(
			learner_id, train_size, validation_size, test_size)


	def register_tfrecords_schemata(self, learner_id, train_schema=None, validation_schema=None, test_schema=None):
		""" For every tfrecord schema we get its items from the fed dictionary and store its string representation. """
		train_schema = str(list(train_schema.items())) if train_schema is not None else train_schema
		validation_schema = str(list(validation_schema.items())) if train_schema is not None else validation_schema
		test_schema = str(list(test_schema.items())) if test_schema is not None else test_schema
		return self.__metis_catalog_client.register_tfrecords_schemata_wcatalog(
			learner_id, train_schema, validation_schema, test_schema)


	def register_tfdatasets_schemata(self, learner_id):
		learner_dataset_client = self.__get_learner_dataset_client(learner_id)
		# TODO By default we register the training dataset schema! This can be changed/extended to support attributes
		#  catalog registration on per dataset type basis.
		train_tfrecords_filepath = self.get_tfrecords_volumes(learner_id, train=True)
		train_tfrecords_schema = self.get_tfrecords_schemata(learner_id, train=True)

		# Make sure that the returned lists are non-empty, so that we can get the head from each list.
		if train_tfrecords_filepath and train_tfrecords_schema:
			filepath, schema = train_tfrecords_filepath[0], train_tfrecords_schema[0]
			# Even if we have non-empty results, we need to make sure that none of the required fields is empty.
			if filepath and schema:
				train_dataset = learner_dataset_client.load_tfrecords(schema, filepath, is_training=True)
				self.__metis_catalog_client.register_tfdataset_schema_wcatalog(
					learner_id, train_dataset.output_types, learner_dataset_client.x_train_input_name(),
					learner_dataset_client.y_train_output_name(), learner_dataset_client.x_eval_input_name(),
					learner_dataset_client.y_eval_output_name())


	def get_federation_learners_ids(self):
		return [dataset_client.learner_id for dataset_client in self.__metisdb_dataset_clients]


	def get_tfrecords_volumes(self, learner_id, train=False, valid=False, test=False):
		return self.__metis_catalog_client.get_learner_tfrecord_volume(learner_id, train, valid, test)


	def get_tfrecords_schemata(self, learner_id, train=False, valid=False, test=False):
		result = self.__metis_catalog_client.get_learner_tfrecord_schema(learner_id, train, valid, test)
		# Upon receiving the tfrecords schemata, we convert each returned list/schema back to an ordered dictionary.
		# The final result is list of ordered dictionaries.
		ordered_tf_schema = TFDatasetUtils.to_ordered_tfschema(result)
		return ordered_tf_schema


	def get_tfrecords_datasizes(self, learner_id, train=False, valid=False, test=False):
		return self.__metis_catalog_client.get_learner_tfrecord_datasize(learner_id, train, valid, test)


	def get_learner_evaluation_attributes(self, learner_id):
		return self.__metis_catalog_client.get_learner_evaluation_attrs(learner_id=learner_id)


	def get_learner_evaluation_input_attribute(self, learner_id):
		return self.__metis_catalog_client.get_learner_evaluation_attrs(learner_id=learner_id)[0]


	def get_learner_evaluation_output_attribute(self, learner_id):
		return self.__metis_catalog_client.get_learner_evaluation_attrs(learner_id=learner_id)[1]


	def import_host_data(self, learner_id, batch_size, import_train=False, import_validation=False, import_test=False):

		learner_dataset_client = self.__get_learner_dataset_client(learner_id)

		# Initialize empty objects for the training, validation and testing datasets.
		train_dataset_ops, valid_dataset_ops, test_dataset_ops = \
			[TFDatasetUtils.structure_tfdataset(None, batch_size, 0)] * 3

		if import_train:
			# TODO Execute a sql query here to fetch worker's data. Query rewriting can be run at this point.
			train_tfrecords_filepath = self.get_tfrecords_volumes(learner_id, train=True)
			train_tfrecords_schema = self.get_tfrecords_schemata(learner_id, train=True)
			train_tfrecords_size = self.get_tfrecords_datasizes(learner_id, train=True)

			# Make sure that the returned lists are non-empty, so that we can get the head from each list.
			if train_tfrecords_filepath and train_tfrecords_schema and train_tfrecords_size:
				filepath, schema, size = train_tfrecords_filepath[0], train_tfrecords_schema[0], train_tfrecords_size[0]
				# Even if we have non-empty results, we need to make sure that none of the required fields is empty.
				if filepath and schema and size:
					train_dataset = learner_dataset_client.load_tfrecords(schema, filepath, is_training=True)
					train_dataset_ops = TFDatasetUtils.structure_tfdataset(train_dataset, batch_size, size,
																		   shuffle=True)

		if import_validation:
			valid_tfrecords_filepath = self.get_tfrecords_volumes(learner_id, valid=True)
			valid_tfrecords_schema = self.get_tfrecords_schemata(learner_id, valid=True)
			valid_tfrecords_size = self.get_tfrecords_datasizes(learner_id, valid=True)

			if valid_tfrecords_filepath and valid_tfrecords_schema and valid_tfrecords_size:
				filepath, schema, size = valid_tfrecords_filepath[0], valid_tfrecords_schema[0], valid_tfrecords_size[0]
				if filepath and schema and size:
					valid_dataset = learner_dataset_client.load_tfrecords(schema, filepath, is_training=False)
					valid_dataset_ops = TFDatasetUtils.structure_tfdataset(valid_dataset, batch_size, size,
																		   shuffle=False)

		if import_test:
			test_tfrecords_filepath = self.get_tfrecords_volumes(learner_id, test=True)
			test_tfrecords_schema = self.get_tfrecords_schemata(learner_id, test=True)
			test_tfrecords_size = self.get_tfrecords_datasizes(learner_id, test=True)

			if test_tfrecords_filepath and test_tfrecords_schema and test_tfrecords_size:
				filepath, schema, size = test_tfrecords_filepath[0], test_tfrecords_schema[0], test_tfrecords_size[0]
				if filepath and schema and size:
					test_dataset = learner_dataset_client.load_tfrecords(schema, filepath, is_training=False)
					test_dataset_ops = TFDatasetUtils.structure_tfdataset(test_dataset, batch_size, size,
																		  shuffle=False)

		return train_dataset_ops, valid_dataset_ops, test_dataset_ops


	def shutdown(self):
		self.__metis_catalog_client.shutdown()