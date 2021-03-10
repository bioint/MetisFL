import re

import tensorflow as tf


class TFConfiguration(object):

	@classmethod
	def tf_session_config(cls, is_gpu=False, per_process_gpu_memory_fraction=1):
		"""
		A protocol buffers configuration function for a running Tensorflow session.
		:return:
		"""

		# TODO device_filters and process_visible_gpus are two ways of explicitly specifying which devices are available
		#  to the current process. However, the error thrown in both cases is: Could not satisfy explicit device
		#  specification '/job:worker/task:2/device:GPU:2' because no supported kernel for GPU devices is available.
		config = tf.ConfigProto(inter_op_parallelism_threads=3, intra_op_parallelism_threads=3)
		config.allow_soft_placement = True
		if is_gpu:
			config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
			config.gpu_options.allow_growth = True

		return config


	@classmethod
	def tf_trainable_variables_config(cls, graph_trainable_variables, user_defined_variables_collection):

		# In case the user fed an other collection as the model's exploration variables. For example, if the user
		# defines MovingAverage variables, we can use them to test the model, then retrieve the collection and finally
		# enforce their correct sequence according to the trainable variables sequence.
		if not any(isinstance(udftv, tf.Variable) for udftv in user_defined_variables_collection):
			raise TypeError("All the trainable variables defined by the user must must be of type %s " % tf.Variable)
		# The following operation sets the correct sequence of the trainable variables by performing a regex matching
		# with the variables collection defined assigned by the user.
		model_vars = [udftv for wtv in graph_trainable_variables for udftv in user_defined_variables_collection
					  if wtv.name in udftv.name or re.sub(r':\d+', '', wtv.name) in udftv.name]

		return model_vars