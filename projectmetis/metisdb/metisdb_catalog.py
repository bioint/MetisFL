import sqlalchemy

import tensorflow as tf

from sqlalchemy.ext.declarative import declarative_base
from metisdb.sqlalchemy_client import SQLAlchemyClient
from sqlalchemy.ext.hybrid import hybrid_method
from utils.generic.singleton import Singleton
from sqlalchemy import *


class MetisCatalog(object, metaclass=Singleton):

	_TF2SQLALCHEMY_DTYPE_MAPPING = {
		tf.int8: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['big_integer'],
		tf.int32: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['big_integer'],
		tf.int64: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['big_integer'],
		tf.float16: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['float'],
		tf.float32: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['float'],
		tf.float64: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['float'],
		tf.bool: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['float'],
		tf.string: SQLAlchemyClient.SQLALCHEMY_DTYPE_INDEX['float'],
	}
	Base = declarative_base()

	def __init__(self, sqlalchemy_client):
		assert isinstance(sqlalchemy_client, SQLAlchemyClient)
		self.sqlalchemy_client = sqlalchemy_client
		# Clear existing catalog, if any.
		self.__delete_catalog_schema()
		# Initialize catalog database.
		self.__generate_catalog_schema()


	class LearnersTFDatasetSchema(Base):
		""" Require value for the model input and output columns. """
		__tablename__ = 'learners_tfdatasets_schemata'
		learner_id = Column(String, primary_key=True)
		created_date = Column(DateTime, primary_key=False)
		x_train_input_name = Column(String, nullable=False)
		x_train_input_dtype = Column(String, nullable=False)
		y_train_output_name = Column(String, nullable=False)
		y_train_output_dtype = Column(String, nullable=False)
		x_eval_input_name = Column(String, nullable=False)
		x_eval_input_dtype = Column(String, nullable=False)
		y_eval_output_name = Column(String, nullable=False)
		y_eval_output_dtype = Column(String, nullable=False)

		@hybrid_method
		def non_dataset_schema_related_columns(self):
			return [self.learner_id, self.created_date]

		@hybrid_method
		def dataset_schema_required_columns(self):
			return [self.x_train_input_name, self.x_train_input_dtype,
					self.y_train_output_name, self.y_train_output_dtype,
					self.x_eval_input_name, self.x_eval_input_dtype,
					self.y_eval_output_name, self.y_eval_output_dtype]

		@classmethod
		def required_schema_insertion_record(cls, learner_id: str,
											 x_train_input_name: str, x_train_input_dtype: str,
											 y_train_output_name: str, y_train_output_dtype: str,
											 x_eval_input_name: str, x_eval_input_dtype: str,
											 y_eval_output_name: str, y_eval_output_dtype: str):
			return {'learner_id': learner_id,
					'created_date': sqlalchemy.func.now(),
					'x_train_input_name': x_train_input_name, 'x_train_input_dtype': x_train_input_dtype,
					'y_train_output_name': y_train_output_name, 'y_train_output_dtype': y_train_output_dtype,
					'x_eval_input_name': x_eval_input_name, 'x_eval_input_dtype': x_eval_input_dtype,
					'y_eval_output_name': y_eval_output_name, 'y_eval_output_dtype': y_eval_output_dtype}


	class LearnersTFRecordVolume(Base):
		""" Require value for the learner's training dataset volume path. """
		__tablename__ = 'learners_tfrecords_volumes'
		learner_id = Column(String, primary_key=True)
		train_tfrecord_volume = Column(String, nullable=False)
		validation_tfrecord_volume = Column(String, nullable=True)
		test_tfrecord_volume = Column(String, nullable=True)

		@classmethod
		def generate_insertion_record(cls, learner_id, train_volume, validation_volume, test_volume):
			return {'learner_id': learner_id,
					'train_tfrecord_volume': train_volume,
					'validation_tfrecord_volume': validation_volume,
					'test_tfrecord_volume': test_volume}


	class LearnersTFRecordSchema(Base):
		""" Require value for the learner's training dataset volume path. """
		__tablename__ = 'learners_tfrecords_schemata'
		learner_id = Column(String, primary_key=True)
		train_tfrecord_schema = Column(String, nullable=False)
		validation_tfrecord_schema = Column(String, nullable=True)
		test_tfrecord_schema = Column(String, nullable=True)

		@classmethod
		def generate_insertion_record(cls, learner_id, train_schema, validation_schema, test_schema):
			return {'learner_id': learner_id,
					'train_tfrecord_schema': train_schema,
					'validation_tfrecord_schema': validation_schema,
					'test_tfrecord_schema': test_schema}


	class LearnersTFRecordSize(Base):
		""" Require value for the learner's training dataset volume path. """
		__tablename__ = 'learners_tfrecords_datasizes'
		learner_id = Column(String, primary_key=True)
		train_tfrecord_size = Column(INTEGER, nullable=False)
		validation_tfrecord_size = Column(INTEGER, nullable=True)
		test_tfrecord_size = Column(INTEGER, nullable=True)

		@classmethod
		def generate_insertion_record(cls, learner_id, train_size, validation_size, test_size):
			return {'learner_id': learner_id,
					'train_tfrecord_size': train_size,
					'validation_tfrecord_size': validation_size,
					'test_tfrecord_size': test_size}


	def __generate_catalog_schema(self):
		self.Base.metadata.tables[self.LearnersTFDatasetSchema.__tablename__].create(bind=self.sqlalchemy_client.dbengine,
																					 checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordVolume.__tablename__].create(bind=self.sqlalchemy_client.dbengine,
																					checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordSchema.__tablename__].create(bind=self.sqlalchemy_client.dbengine,
																					checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordSize.__tablename__].create(bind=self.sqlalchemy_client.dbengine,
																				  checkfirst=True)

	def __delete_catalog_schema(self):
		self.Base.metadata.tables[self.LearnersTFDatasetSchema.__tablename__].drop(bind=self.sqlalchemy_client.dbengine,
																				   checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordVolume.__tablename__].drop(bind=self.sqlalchemy_client.dbengine,
																				  checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordSchema.__tablename__].drop(bind=self.sqlalchemy_client.dbengine,
																				  checkfirst=True)
		self.Base.metadata.tables[self.LearnersTFRecordSize.__tablename__].drop(bind=self.sqlalchemy_client.dbengine,
																				checkfirst=True)

	def __tf2sqlalchemy_dtype(self, tf_dtype: str):
		if tf_dtype in self._TF2SQLALCHEMY_DTYPE_MAPPING:
			return self._TF2SQLALCHEMY_DTYPE_MAPPING[tf_dtype]
		else:
			raise NotImplementedError("{} DataType not yet supported.".format(str(tf_dtype)))


	def __convert_to_sqlalchemy_schema(self, input_schema: dict):
		sqlalchemy_schema = {}
		for feature_name, tf_dtype in input_schema.items():
			sqlalchemy_schema[feature_name] = self.__tf2sqlalchemy_dtype(tf_dtype)
		return sqlalchemy_schema


	def __add_table_column(self, table_name: str, column: Column):
		column_name = column.compile(dialect=self.sqlalchemy_client.dbengine.dialect)
		column_type = column.type.compile(self.sqlalchemy_client.dbengine.dialect)

		if self.sqlalchemy_client.is_postgres_instance:
			self.sqlalchemy_client.dbengine.execute("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {}"
										   .format(table_name, column_name, column_type))
		elif self.sqlalchemy_client.is_sqlite_instance:
			# Unfortunately sqlite does not support 'IF NOT EXISTS' SQL clauses, so we need to try-catch.
			try:
				self.sqlalchemy_client.dbengine.execute("ALTER TABLE {} ADD COLUMN {} {}"
													.format(table_name, column_name, column_type))
			except sqlalchemy.exc.OperationalError as exc:
				pass


	def register_tfrecords_volumes_wcatalog(self, learner_id, train_tfrecord_volume=None,
											validation_tfrecord_volume=None, test_tfrecord_volume=None):
		assert any([train_tfrecord_volume is not None, validation_tfrecord_volume is not None,
					test_tfrecord_volume is not None]), "Please indicate at least one volume path!"

		insertion_record = self.LearnersTFRecordVolume.generate_insertion_record(learner_id,
																				 train_tfrecord_volume,
																				 validation_tfrecord_volume,
																				 test_tfrecord_volume)
		insertion_table = Table(self.LearnersTFRecordVolume.__tablename__,
								self.Base.metadata,
								autoload_with=self.sqlalchemy_client.dbengine,
								extend_existing=True)
		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordVolume.learner_id == learner_id)).scalar()

		if pk_filter_exists:
			current_session().execute(
				update(self.LearnersTFRecordVolume, values={
					self.LearnersTFRecordVolume.train_tfrecord_volume: train_tfrecord_volume,
					self.LearnersTFRecordVolume.validation_tfrecord_volume: validation_tfrecord_volume,
					self.LearnersTFRecordVolume.test_tfrecord_volume: test_tfrecord_volume})
				.where(self.LearnersTFRecordVolume.learner_id == learner_id))

		else:
			insert_statement = sqlalchemy.sql.insert(insertion_table)
			sql_statement = insert_statement.values(insertion_record)
			self.sqlalchemy_client.dbengine.execute(sql_statement)
		current_session.close()

		return insertion_record


	def register_tfrecords_schemata_wcatalog(self, learner_id, train_tfrecords_schema,
											 validation_tfrecords_schema=None, test_tfrecords_schema=None):

		insertion_record = self.LearnersTFRecordSchema.generate_insertion_record(learner_id,
																				 train_tfrecords_schema,
																				 validation_tfrecords_schema,
																				 test_tfrecords_schema)
		insertion_table = Table(self.LearnersTFRecordSchema.__tablename__,
								self.Base.metadata,
								autoload_with=self.sqlalchemy_client.dbengine,
								extend_existing=True)
		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordSchema.learner_id == learner_id)).scalar()

		if pk_filter_exists:
			current_session.execute(
				update(self.LearnersTFRecordSchema, values={
					self.LearnersTFRecordSchema.train_tfrecord_schema: train_tfrecords_schema,
					self.LearnersTFRecordSchema.validation_tfrecord_schema: validation_tfrecords_schema,
					self.LearnersTFRecordSchema.test_tfrecord_schema: test_tfrecords_schema})
				.where(self.LearnersTFRecordSchema.learner_id == learner_id))

		else:
			insert_statement = sqlalchemy.sql.insert(insertion_table)
			sql_statement = insert_statement.values(insertion_record)
			self.sqlalchemy_client.dbengine.execute(sql_statement)
		current_session.close()

		return insertion_record


	def register_tfrecords_sizes_wcatalog(self, learner_id, train_size=0, validation_size=0, test_size=0):

		insertion_record = self.LearnersTFRecordSize.generate_insertion_record(learner_id,
																			   train_size,
																			   validation_size,
																			   test_size)
		insertion_table = Table(self.LearnersTFRecordSize.__tablename__,
								self.Base.metadata,
								autoload_with=self.sqlalchemy_client.dbengine,
								extend_existing=True)
		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordSize.learner_id == learner_id)).scalar()

		if pk_filter_exists:
			current_session.execute(
				update(self.LearnersTFRecordSize, values={
					self.LearnersTFRecordSize.train_tfrecord_size: train_size,
					self.LearnersTFRecordSize.validation_tfrecord_size: validation_size,
					self.LearnersTFRecordSize.test_tfrecord_size: test_size})
				.where(self.LearnersTFRecordSize.learner_id == learner_id))

		else:
			insert_statement = sqlalchemy.sql.insert(insertion_table)
			sql_statement = insert_statement.values(insertion_record)
			self.sqlalchemy_client.dbengine.execute(sql_statement)
		current_session.close()

		return insertion_record


	def register_tfdataset_schema_wcatalog(self, learner_id: str, input_schema: dict,
										   x_train_input_attr_name: str, y_train_output_attr_name: str,
										   x_eval_input_attr_name: str, y_eval_output_attr_name: str):
		"""
		A helper function to register the schema of a Tensroflow record (.tf). The input schema must be
		a dictionary with keys being the name of the attribute and value the Tensroflow type (tf.DType).
		Given that some datasets might follow a different schema, the function processes every attribute
		and if the number of attributes in the database table are less in number than the ones passed
		through the input schema, then the function extends the existing `learners_dataset_schema` definition
		table to accommodate the additional attributes.
		:param learner_id:
		:param input_schema:
		:param x_train_input_attr_name:
		:param y_train_output_attr_name:
		:param x_eval_input_attr_name:
		:param y_eval_output_attr_name:
		:return:
		"""

		assert isinstance(input_schema, dict)
		for tf_dtype in input_schema.values():
			assert isinstance(tf_dtype, tf.dtypes.DType)

		# Make sure input and output attributes are different.
		assert x_train_input_attr_name != y_train_output_attr_name, "X training input same as Y training output."
		assert x_eval_input_attr_name != y_eval_output_attr_name, "X evaluation input same as Y evaluation output."

		# Extract information for the required input and output columns.
		# Extract the key of the model input and output and its current value.
		# The rational is to be able to register the additional features a
		# new dataset schema might introduce.

		# Training features.
		x_train_input_name = x_train_input_attr_name
		x_train_input_dtype = input_schema.pop(x_train_input_name).name # Get the DType() actual name, e.g. tf.int64 as int64
		y_train_output_name = y_train_output_attr_name
		y_train_output_dtype = input_schema.pop(y_train_output_name).name # Get the DType() actual name.

		# Evaluation features. If same as input training features then map accordingly. X input.
		if x_train_input_attr_name == x_eval_input_attr_name:
			x_eval_input_name = x_train_input_name
			x_eval_input_dtype = x_train_input_dtype
		else:
			x_eval_input_name = x_eval_input_attr_name
			x_eval_input_dtype = input_schema.pop(x_eval_input_name).name # Get the DType() actual name, e.g. tf.int64 as int64

		# Evaluation features. Y output.
		if y_train_output_attr_name == y_eval_output_attr_name:
			y_eval_output_name = y_train_output_name
			y_eval_output_dtype = y_train_output_dtype
		else:
			y_eval_output_name = y_eval_output_attr_name
			y_eval_output_dtype = input_schema.pop(y_eval_output_name).name

		insertion_record = self.LearnersTFDatasetSchema.required_schema_insertion_record(learner_id,
																						 x_train_input_name, x_train_input_dtype,
																						 y_train_output_name, y_train_output_dtype,
																						 x_eval_input_name, x_eval_input_dtype,
																						 y_eval_output_name, y_eval_output_dtype)

		# After popping out the required columns from the input schema, we can safely assume
		# that the remaining schema columns are just extra features. So we insert a value to
		# every feature in a round-robin fashion. There is no particular ordering for the extra features!
		for feature_idx, items in enumerate(sorted(input_schema.items()), start=1):
			feature_name = items[0] # dictionary key.
			feature_dtype = items[1].name # dictionary value, get the DType's string name.

			column_name = "Extra_Feature_{}_name".format(feature_idx)
			sqlalchemy_column = Column(String, name=column_name, nullable=False)
			self.__add_table_column(self.LearnersTFDatasetSchema.__tablename__, sqlalchemy_column)
			insertion_record[column_name] = feature_name

			column_name = "Extra_Feature_{}_dtype".format(feature_idx)
			sqlalchemy_column = Column(String, name=column_name, nullable=False)
			self.__add_table_column(self.LearnersTFDatasetSchema.__tablename__, sqlalchemy_column)
			insertion_record[column_name] = feature_dtype

		insertion_table = Table(self.LearnersTFDatasetSchema.__tablename__,
								self.Base.metadata,
								autoload_with=self.sqlalchemy_client.dbengine,
								extend_existing=True)
		insert_statement = sqlalchemy.sql.insert(insertion_table)
		insert_statement = insert_statement.values(insertion_record)
		# Try-catch and rollback in case the learner has already registered its dataset.
		# In this case update learner's existing record with new values (delete & insert).
		try:
			self.sqlalchemy_client.dbengine.execute(insert_statement)
		except sqlalchemy.exc.IntegrityError:
			current_session = self.sqlalchemy_client.session()
			current_session.rollback()
			delete_learner_record = insertion_table.delete().where(
				self.LearnersTFDatasetSchema.learner_id == learner_id)
			self.sqlalchemy_client.dbengine.execute(delete_learner_record)
			self.sqlalchemy_client.dbengine.execute(insert_statement)
			current_session.close()

		return insertion_record


	def get_learner_tfrecord_volume(self, learner_id, train=False, valid=False, test=False):

		assert any([train, valid, test]), "Need to specify one volume type."

		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordVolume.learner_id == learner_id)).scalar()
		val = None
		if pk_filter_exists:
			# Set the volume columns to project.
			projection_columns = []
			if train:
				projection_columns.append(self.LearnersTFRecordVolume.train_tfrecord_volume)
			if valid:
				projection_columns.append(self.LearnersTFRecordVolume.validation_tfrecord_volume)
			if test:
				projection_columns.append(self.LearnersTFRecordVolume.test_tfrecord_volume)
			val = current_session\
				.query(*projection_columns)\
				.filter(self.LearnersTFRecordVolume.learner_id == learner_id)\
				.all()
			val = val[0]
		current_session.close()
		return val


	def get_learner_tfrecord_schema(self, learner_id, train=False, valid=False, test=False):

		assert any([train, valid, test]), "Need to specify at least one volume type."

		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordSchema.learner_id == learner_id)).scalar()
		val = None
		if pk_filter_exists:
			# Set the schema columns to project.
			projection_columns = []
			if train:
				projection_columns.append(self.LearnersTFRecordSchema.train_tfrecord_schema)
			if valid:
				projection_columns.append(self.LearnersTFRecordSchema.validation_tfrecord_schema)
			if test:
				projection_columns.append(self.LearnersTFRecordSchema.test_tfrecord_schema)

			val = current_session\
				.query(*projection_columns)\
				.filter(self.LearnersTFRecordSchema.learner_id == learner_id)\
				.all()
			val = val[0]
		current_session.close()
		return val


	def get_learner_tfrecord_datasize(self, learner_id, train=False, valid=False, test=False):

		assert any([train, valid, test]), "Need to specify at least one data type."

		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFRecordSize.learner_id == learner_id)).scalar()
		val = None
		if pk_filter_exists:
			# Set the schema columns to project.
			projection_columns = []
			if train:
				projection_columns.append(self.LearnersTFRecordSize.train_tfrecord_size)
			if valid:
				projection_columns.append(self.LearnersTFRecordSize.validation_tfrecord_size)
			if test:
				projection_columns.append(self.LearnersTFRecordSize.test_tfrecord_size)

			val = current_session\
				.query(*projection_columns)\
				.filter(self.LearnersTFRecordSize.learner_id == learner_id)\
				.all()
			val = val[0]
		current_session.close()
		return val


	def get_learner_training_attrs(self, learner_id):
		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFDatasetSchema.learner_id == learner_id)).scalar()
		val = None
		if pk_filter_exists:
			val = current_session\
				.query(self.LearnersTFDatasetSchema.x_train_input_name,
					   self.LearnersTFDatasetSchema.y_train_output_name)\
				.filter(self.LearnersTFDatasetSchema.learner_id == learner_id)\
				.all()
			# Val is of type list. The two training attributes are returned in a tuple format.
			val = val[0]
		current_session.close()
		return val


	def get_learner_evaluation_attrs(self, learner_id):
		current_session = self.sqlalchemy_client.session()
		pk_filter_exists = current_session.query(exists().where(
			self.LearnersTFDatasetSchema.learner_id == learner_id)).scalar()
		val = None
		if pk_filter_exists:
			val = current_session\
				.query(self.LearnersTFDatasetSchema.x_eval_input_name,
					   self.LearnersTFDatasetSchema.y_eval_output_name)\
				.filter(self.LearnersTFDatasetSchema.learner_id == learner_id)\
				.all()
			# Val is of type list. The two evaluation attributes are returned in a tuple format.
			val = val[0]
		current_session.close()
		return val


	def shutdown(self):
		self.__delete_catalog_schema()
		self.sqlalchemy_client.session().close()