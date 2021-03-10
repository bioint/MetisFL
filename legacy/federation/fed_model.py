import abc

import numpy as np
import tensorflow as tf

from utils.logging.metis_logger import MetisLogger as metis_logger


class FedTensor(object):

	def __init__(self, tensor, feed_dict):
		self.__tensor = tensor
		self.__feed_dict = feed_dict

	def eval_tf_tensor(self, session, extra_feeds={}):
		self.__feed_dict.update(extra_feeds)
		return self.__tensor.eval(feed_dict=self.__feed_dict, session=session)

	def get_feed_dictionary(self):
		return self.__feed_dict

	def get_tensor(self):
		return self.__tensor


class FedOperation(object):

	def __init__(self, operation, feed_dict):
		self.__operation = operation
		self.__feed_dict = feed_dict

	def run_tf_operation(self, session, extra_feeds={}):
		self.__feed_dict.update(extra_feeds)
		return self.__operation.run(feed_dict=self.__feed_dict, session=session)

	def get_feed_dictionary(self):
		return self.__feed_dict

	def get_operation(self):
		return self.__operation


class FedVar(object):

	def __init__(self, name, value, dtype=None, shape=None, trainable=True):
		if name is None:
			raise ValueError("`name` argument cannot be empty, a name must be defined for the federation variable")
		self.name = name
		self.value = value
		self.dtype = dtype
		self.shape = shape
		self.trainable = trainable
		self.tf_collection = [tf.GraphKeys.GLOBAL_VARIABLES]


class ModelArchitecture(object):

	def __init__(self, loss, train_step, predictions, model_federated_variables, accuracy=None,
				 user_defined_testing_variables_collection=None, **kwargs):
		if not isinstance(train_step, FedOperation):
			raise TypeError("`train_step` must be of type %s" % FedOperation)
		if not isinstance(predictions, FedTensor):
			raise TypeError("`accuracy` must be of type %s" % FedTensor)
		self.loss = loss
		self.train_step = train_step
		self.predictions = predictions
		self.model_federated_variables = model_federated_variables
		self.accuracy = accuracy
		self.user_defined_testing_variables_collection = user_defined_testing_variables_collection
		self.kwargs = kwargs


class FedModelDef(abc.ABC):

	@property
	@abc.abstractmethod
	def input_tensors_datatype(self):
		pass


	@property
	@abc.abstractmethod
	def output_tensors_datatype(self):
		pass


	@abc.abstractmethod
	def model_architecture(self, input_tensors, output_tensors, global_step, batch_size, dataset_size, **kwargs) \
			-> ModelArchitecture:

		"""
		Function which specifies the entire architecture of the network.
		:param input_tensors: the input data(tensors) to the network
		:param output_tensors: the labels of the input data
		:param global_step:
		:param batch_size:
		:param dataset_size:

		:return: :type: ModelArchitecture
		"""
		pass


	@staticmethod
	def construct_model_federated_variables(federation_model_obj) -> [FedVar]:
		"""
		This is a core function for the federation environment. It takes as an input
		the newly created federation model and duplicates each variable marked as a
		federation variable (parameter model_federated_variables) in the given model
		by renaming it using the prefix `federated_variable_`.

		We use this renaming in order to register them as new variables inside the learner's
		local model under a designated tensorflow scope.

		Args:
			federation_model_obj:

		Returns:

		"""

		if not isinstance(federation_model_obj, FedModelDef):
			raise TypeError("The `federation_model_obj` parameter must be of type %s " % FedModelDef)

		run_meta = tf.RunMetadata()
		federated_variables = list()
		with tf.Graph().as_default() as graph:

			metis_logger.info("Initializing Federated Variables...")
			global_step = tf.train.get_or_create_global_step()
			model_architecture = federation_model_obj.model_architecture(
				input_tensors=federation_model_obj.input_tensors_datatype(),
				output_tensors=federation_model_obj.output_tensors_datatype(),
				global_step=global_step, batch_size=None, dataset_size=None)
			opts = tf.profiler.ProfileOptionBuilder.float_operation()
			flops = tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)
			metis_logger.info(msg="Federated Model FLOPs: {}".format(flops.total_float_ops))

			trainable_vars = model_architecture.model_federated_variables
			model_number_of_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in trainable_vars])

			fedvar_name_prefix = "federated_variable_{}_{}"
			network_size = 0
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				trainable_vars_values = sess.run(trainable_vars)
				for var_idx, var_val in enumerate(trainable_vars_values):

					fedarray_mb = np.divide(np.divide(var_val.nbytes, 1024), 1024) # MBs
					network_size += fedarray_mb
					# TODO This name substitution might not be stable in subsequent tensorflow releases
					fedvar_name_suffix = trainable_vars[var_idx].op.name.replace("/", "_")
					fedvar_dtype = trainable_vars[var_idx].dtype
					fedvar_shape = trainable_vars[var_idx].shape
					fedvar_name = fedvar_name_prefix.format(var_idx, fedvar_name_suffix)
					fedvar = FedVar(name=fedvar_name, value=var_val, dtype=fedvar_dtype, shape=fedvar_shape)
					federated_variables.append(fedvar)
			metis_logger.info(msg="Federated Variables Initialized")
			metis_logger.info(msg="Federated Model Size: {0:.2f} MB".format(round(network_size, 2)))
			metis_logger.info(msg="Federated Model Number of Parameters: {}".format(model_number_of_parameters))

		return federated_variables
