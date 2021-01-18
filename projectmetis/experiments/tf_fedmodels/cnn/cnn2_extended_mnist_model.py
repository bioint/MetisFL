import federation.fed_model as fedmodel
import tensorflow as tf

class ExtendedMnistFedModel(fedmodel.FedModelDef):
	"""
	An example use case of how to define the federation model and the associated federation variables, by implementing
	the two functions `model_variables_structure()` and `model_architecture()` of `FedModelDef` base class
	"""

	def __init__(self, learning_rate=0.01, momentum=0.0, num_classes=62):
		self.initial_learning_rate = learning_rate
		self.sgd_momentum = momentum
		self.num_classes = num_classes

	def input_tensors_datatype(self, **kwargs):
		return { 'images': tf.placeholder(tf.float32, [None, 28, 28, 1], name='images') }

	def output_tensors_datatype(self, **kwargs):
		return { 'labels': tf.placeholder(tf.int64, [None], name='labels') }

	def model_architecture(self, input_tensors, output_tensors, global_step=None, batch_size=None, dataset_size=None, **kwargs):
		"""
		:param input_tensors: the input data to the network
		:param output_tensors: output data of the network
		:param global_step
		:param batch_size
		:param dataset_size

		:return:
		"""

		x_images = input_tensors['images']
		output_tensors = output_tensors['labels']

		# Need to cast the input to tf.float64, since during training the
		# trainable weights get altered to float64 type.
		x_images = tf.cast(x_images, dtype=tf.float64)

		with tf.name_scope('conv1'):
			conv1 = tf.layers.conv2d(
				inputs=x_images,
				filters=32,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu)

		with tf.name_scope('pool1'):
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

		with tf.name_scope('conv2'):
			conv2 = tf.layers.conv2d(
				inputs=pool1,
				filters=64,
				kernel_size=[5, 5],
				padding="same",
				activation=tf.nn.relu)

		with tf.name_scope('pool2'):
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
			pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
			dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)

		with tf.name_scope('predictions'):
			logits = tf.layers.dense(inputs=dense, units=self.num_classes)
			predictions = {
				"classes": tf.argmax(input=logits, axis=1),
				"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
			}

		with tf.name_scope('evaluation'):
			eval_metric_ops = tf.count_nonzero(tf.equal(output_tensors, predictions["classes"]))

		with tf.name_scope('loss'):
			loss = tf.losses.sparse_softmax_cross_entropy(labels=output_tensors, logits=logits)

		def training_op(loss):
			# lr_annealing_value = tf.placeholder(tf.float32, name="lr_annealing_value")
			# momentum_annealing_value = tf.placeholder(tf.float32, name="momentum_annealing_value")
			# lr_value = tf.multiply(self.initial_learning_rate, 1-lr_annealing_value, name="lr_value")
			# momentum_value = tf.multiply(self.sgd_momentum, 1+momentum_annealing_value, name="momentum_value")
			optimizer = tf.train.MomentumOptimizer(
				learning_rate=self.initial_learning_rate,
				momentum=self.sgd_momentum
			)
			train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_global_step())
			return train_op


		# Return Model Config Settings
		train_op = training_op(loss)
		predictions_tensor = predictions['classes']

		fed_train_step_op = fedmodel.FedOperation(operation=train_op, feed_dict={})
		fed_predictions = fedmodel.FedTensor(tensor=predictions_tensor, feed_dict={})
		fed_accuracy_tensor = fedmodel.FedTensor(tensor=eval_metric_ops, feed_dict={})
		fed_loss_tensor = fedmodel.FedTensor(tensor=loss, feed_dict={})
		fed_trainable_variables_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		return fedmodel.ModelArchitecture(train_step=fed_train_step_op,
										  predictions=fed_predictions,
										  accuracy=fed_accuracy_tensor,
										  loss=fed_loss_tensor,
										  model_federated_variables=fed_trainable_variables_collection)