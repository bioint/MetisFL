import federation.fed_model as fedmodel
import tensorflow as tf


# NOTE The following implementation of mnist_deep_CNN is largely based on: https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/mnist/mnist_deep.py
# tf.logging.set_verbosity(4)  # Enable INFO verbosity

class MnistFedModel(fedmodel.FedModelDef):
	"""
	An example use case of how to define the federation model and the associated federation variables, by implementing
	the two functions `model_variables_structure()` and `model_architecture()` of `FedModelDef` base class
	"""

	def __init__(self, learning_rate=0.0015, momentum=0.0):
		self.initial_learning_rate = learning_rate
		self.sgd_momentum = momentum

	def input_tensors_datatype(self, **kwargs):
		return { 'images': tf.placeholder(tf.float32, [None, 28, 28, 1], name='images') }

	def output_tensors_datatype(self, **kwargs):
		return { 'labels': tf.placeholder(tf.int64, [None], name='labels') }

	def model_architecture(self, input_tensors, output_tensors, class_ids_mask=None,
						   global_step=None, batch_size=None, dataset_size=None, **kwargs):
		"""
		:param input_tensors: the input data to the network
		:param output_tensors: output data of the network
		:param global_step
		:param batch_size
		:param dataset_size

		:return:
		"""

		def conv2d(x, W):
			"""conv2d returns a 2d convolution layer with full stride."""
			return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

		def max_pool_2x2(x):
			"""max_pool_2x2 downsamples a feature map by 2X."""
			return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		x_images = input_tensors['images']
		output_labels = output_tensors['labels']

		# First convolutional layer -- maps one grayscale image to 32 feature maps.
		with tf.name_scope('conv1'):
			with tf.variable_scope('conv1_vars'):
				W_conv1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1), name='weight_conv1')
				b_conv1 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[32]), name='bias_conv1')
			h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)

		# Pooling layer -- downsamples by 2X.
		with tf.name_scope('pool1'):
			h_pool1 = max_pool_2x2(h_conv1)

		# Second convolutional layer -- maps 32 feature maps to 64.
		with tf.name_scope('conv2'):
			with tf.variable_scope('conv2_vars'):
				W_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1), name='weight_conv2')
				b_conv2 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[64]), name='bias_conv2')
			h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

		# Second pooling layer.
		with tf.name_scope('pool2'):
			h_pool2 = max_pool_2x2(h_conv2)

		# Fully connected layer 1 - after 2 rounds of downsampling, our 28x28 image
		# is down to 7x7x64 feature maps -- maps this to 512 features.
		with tf.name_scope('fc1'):
			with tf.variable_scope('fc1_vars'):
				W_fc1 = tf.Variable(initial_value=tf.truncated_normal(shape=[7 * 7 * 64, 512], stddev=0.1), name='weight_fc1')
				b_fc1 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[512]), name='bias_fc1')
			h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Map the 1024 features to 10 classes, one for each digit
		with tf.name_scope('fc2'):
			with tf.variable_scope('fc2_vars'):
				W_fc2 = tf.Variable(initial_value=tf.truncated_normal(shape=[512, 10], stddev=0.1), name='weight_fc2')
				b_fc2 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[10]), name='bias_fc2')
			y_conv = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name='y_conv')
			if class_ids_mask is not None:
				# class_ids_mask = tf.constant([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], name='class_ids_mask')
				y_conv = tf.multiply(y_conv, class_ids_mask, name='y_conv_with_cid_mask')

		# Define Loss
		with tf.name_scope('loss'):
			cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=output_labels, logits=y_conv)
		cross_entropy = tf.reduce_mean(cross_entropy)

		# Define Optimizer
		lr_annealing_value = tf.placeholder(tf.float32, name="lr_annealing_value")
		momentum_annealing_value = tf.placeholder(tf.float32, name="momentum_annealing_value")
		lr_value = tf.multiply(self.initial_learning_rate, 1 - lr_annealing_value, name="lr_value")
		momentum_value = tf.multiply(self.sgd_momentum, 1 + momentum_annealing_value, name="momentum_value")
		with tf.name_scope('optimizer'):
			train_op = tf.train.AdamOptimizer(learning_rate=lr_value).minimize(cross_entropy)
			# train_op = tf.train.GradientDescentOptimizer(initial_learning_rate=self.initial_learning_rate).minimize(cross_entropy)

		with tf.name_scope('predictions'):
			predictions_tensor = tf.argmax(y_conv, 1)

		# Define Accuracy
		with tf.name_scope('accuracy'):
			correct_predictions = tf.equal(predictions_tensor, output_labels)
			correct_predictions = tf.cast(correct_predictions, tf.float32)
		accuracy_tensor = tf.reduce_mean(correct_predictions)

		# Return Model Config Settings
		# return train_step, accuracy and model trainable variables
		fed_train_step_op = fedmodel.FedOperation(operation=train_op, feed_dict={})
		fed_predictions = fedmodel.FedTensor(tensor=predictions_tensor, feed_dict={})
		fed_accuracy_tensor = fedmodel.FedTensor(tensor=accuracy_tensor, feed_dict={})
		fed_loss_tensor = fedmodel.FedTensor(tensor=cross_entropy, feed_dict={})
		fed_trainable_variables_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		return fedmodel.ModelArchitecture(train_step=fed_train_step_op,
										  predictions=fed_predictions,
										  accuracy=fed_accuracy_tensor,
										  loss=fed_loss_tensor,
										  model_federated_variables=fed_trainable_variables_collection)

