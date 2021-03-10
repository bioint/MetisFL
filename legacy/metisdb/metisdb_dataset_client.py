import tensorflow as tf
import collections
import abc


class MetisDatasetClient(abc.ABC):


	def __init__(self, learner_id):
		self.learner_id = learner_id


	@abc.abstractmethod
	def generate_tfrecords(self, *args, **kwargs) -> (int, collections.OrderedDict):
		""" An abstract method that generates the .tfrecords from each learner's local dataset and returns
			the number of examples that were serialized into tfrecords and the tfrecords schema as ordered dictionary"""
		pass


	@abc.abstractmethod
	def load_tfrecords(self, tfrecords_schema: dict, tfrecords_filepath: str, is_training: bool) -> \
			tf.data.TFRecordDataset:
		pass


	@abc.abstractmethod
	def x_train_input_name(self) -> str:
		pass


	@abc.abstractmethod
	def y_train_output_name(self) -> str:
		pass


	@abc.abstractmethod
	def x_eval_input_name(self) -> str:
		pass


	@abc.abstractmethod
	def y_eval_output_name(self) -> str:
		pass