import experiments.tf_fedmodels.resnet.resnet_model as resnet_model
import federation.fed_model as fedmodel
import tensorflow as tf

class ResNetCifarFedModel(fedmodel.FedModelDef):

	def __init__(self, num_classes, learning_rate, momentum, data_format=None,
				 resnet_version=resnet_model.DEFAULT_VERSION, weight_decay=2e-4,
				 loss_scale=1, resnet_size=50, run_with_distorted_images=True, dtype=tf.float32):

		if resnet_size % 6 != 2:
			raise ValueError('resnet_size must be 6n + 2:', resnet_size)

		self.num_blocks = (resnet_size - 2) // 6
		self.num_classes = num_classes

		self.learning_rate = learning_rate
		self.momentum = momentum

		# Parameter data format can be: None, channels_first, channels_last
		# channels_first provides a performance boost on GPU but is not
		# always compatible with CPU. If left unspecified, the data format
		# will be chosen automatically based on whether TensorFlow was
		# built for CPU or GPU."
		self.data_format = data_format

		self.resnet_size = resnet_size
		self.resnet_version = resnet_version

		# We use a weight decay of 0.0002, which performs better
		# than the 0.0001 that was originally suggested.
		self.weight_decay = weight_decay

		self.loss_scale = loss_scale
		self.dtype = dtype

		self.resnet_model = resnet_model.Model(
			resnet_size=resnet_size,
			bottleneck=False,
			num_classes=self.num_classes,
			num_filters=16,
			kernel_size=3,
			conv_stride=1,
			first_pool_size=None,
			first_pool_stride=None,
			block_sizes=[self.num_blocks] * 3,
			block_strides=[1, 2, 2],
			final_size=64,
			resnet_version=self.resnet_version,
			data_format=self.data_format,
			dtype=self.dtype
		)

		self.run_with_distorted_images = run_with_distorted_images
		self.img_depth = 3
		if self.run_with_distorted_images:
			self.img_height = 24
			self.img_width = 24
		else:
			self.img_height = 32
			self.img_width = 32

	def learning_rate_with_decay(self,
			batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
		"""Get a learning rate that decays step-wise as training progresses.

		Args:
		  batch_size: the number of examples processed in each training batch.
		  batch_denom: this value will be used to scale the base learning rate.
			`0.1 * batch size` is divided by this number, such that when
			batch_denom == batch_size, the initial learning rate will be 0.1.
		  num_images: total number of images that will be used for training.
		  boundary_epochs: list of ints representing the epochs at which we
			decay the learning rate.
		  decay_rates: list of floats representing the decay rates to be used
			for scaling the learning rate. It should have one more element
			than `boundary_epochs`, and all elements should have the same type.

		Returns:
		  Returns a function that takes a single argument - the number of batches
		  trained so far (global_step)- and returns the learning rate to be used
		  for training the next batch.
		"""
		initial_learning_rate = 0.1 * batch_size / batch_denom
		batches_per_epoch = num_images / batch_size

		# Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
		boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
		vals = [initial_learning_rate * decay for decay in decay_rates]

		def learning_rate_fn(global_step):
			global_step = tf.cast(global_step, tf.int32)
			return tf.train.piecewise_constant(global_step, boundaries, vals)

		return learning_rate_fn


	def input_tensors_datatype(self):
		return {'images': tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_depth])}


	def output_tensors_datatype(self):
		return {'labels': tf.placeholder(tf.int64, [None])}


	def model_architecture(self, input_tensors, output_tensors, global_step, batch_size, **kwargs):

		input = input_tensors['images']
		output = output_tensors['labels']

		def inference(images):
			logits = self.resnet_model(inputs=images, training=tf.placeholder(tf.bool, name='is_training_op'))
			# This acts as a no-op if the logits are already in fp32 (provided logits are
			# not a SparseTensor). If dtype is low precision, logits must be cast to
			# fp32 for numerical stability.
			logits = tf.cast(logits, tf.float32)
			return logits

		def predictions(logits):
			classes = tf.argmax(logits, axis=1),
			probabilities = tf.nn.softmax(logits, name='softmax_tensor')
			return classes, probabilities

		def loss(logits, labels):
			# Calculate loss, which includes: 1. softmax cross entropy and 2. L2 regularization.
			cross_entropy = tf.losses.sparse_softmax_cross_entropy(
				logits=logits, labels=labels)
			# Create a tensor named cross_entropy for logging purposes.
			tf.identity(cross_entropy, name='cross_entropy')
			# tf.summary.scalar('cross_entropy', cross_entropy)

			# If no loss_filter_fn is passed, assume we want the default behavior,
			# which is that batch_normalization variables are excluded from loss.
			def exclude_batch_norm(name):
				return 'batch_normalization' not in name

			def loss_filter_fn(_):
				return True

			# Always include batch normalization in Cifar10
			loss_filter_fn = loss_filter_fn or exclude_batch_norm

			# Add weight decay (L2 Regularization) to the loss.
			l2_loss = self.weight_decay * tf.add_n(
				# loss is computed using fp32 for numerical stability.
				[tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
				 if loss_filter_fn(v.name)])
			# tf.summary.scalar('l2_loss', l2_loss)
			loss = cross_entropy + l2_loss

			return loss

		def train(total_loss):

			# lr_annealing_value = tf.placeholder(tf.float32, name="lr_annealing_value")
			# momentum_annealing_value = tf.placeholder(tf.float32, name="momentum_annealing_value")
			# lr_value = tf.multiply(self.learning_rate, 1 - lr_annealing_value, name="lr_value")
			# momentum_value = tf.multiply(self.momentum, 1 + momentum_annealing_value, name="momentum_value")

			# learning_rate_fn = self.learning_rate_with_decay(
			# 	batch_size=128, batch_denom=128,
			# 	num_images=50000, boundary_epochs=[100, 150, 200],
			# 	decay_rates=[1, 0.1, 0.01, 0.001])
			# learning_rate_value = learning_rate_fn(global_step)
			# tf.identity(learning_rate_value, name='learning_rate_value')
			# tf.identity(global_step, name='global_step_value')

			optimizer = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate,
				momentum=self.momentum
			)

			if self.loss_scale != 1:
				# When computing fp16 gradients, often intermediate tensor values are
				# so small, they underflow to 0. To avoid this, we multiply the loss by
				# loss_scale to make these tensor values loss_scale times bigger.
				scaled_grad_vars = optimizer.compute_gradients(total_loss * self.loss_scale)

				# Once the gradient computation is complete we can scale the gradients
				# back to the correct scale before passing them to the optimizer.
				unscaled_grad_vars = [(grad / self.loss_scale, var)
									  for grad, var in scaled_grad_vars]
				minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
			else:
				minimize_op = optimizer.minimize(total_loss, global_step)

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			train_op = tf.group(minimize_op, update_ops)
			return train_op

		def accuracy_value(logits):
			"""
				This function is not needed now, since accuracy is computed through Metis using the confusion matrix
			"""
			classes, probabilities = predictions(logits=logits)
			accuracy = tf.metrics.accuracy(output, classes)
			return accuracy

		logits = inference(input)
		classes, probabilities = predictions(logits)
		predictions_tensor = classes
		total_loss_tensor = loss(logits, output)
		train_op = train(total_loss_tensor)

		fed_loss_tensor = fedmodel.FedTensor(tensor=total_loss_tensor, feed_dict={'is_training_op:0': False})
		fed_train_op = fedmodel.FedOperation(operation=train_op, feed_dict={'is_training_op:0': True})
		fed_predictions = fedmodel.FedTensor(tensor=predictions_tensor, feed_dict={'is_training_op:0': False})
		# federated_variables_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		federated_variables_collection = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
										  	('kernel' in var.name and 'Momentum' not in var.name) or
										  	('gamma' in var.name and 'Momentum' not in var.name) or
											('beta' in var.name and 'Momentum' not in var.name) or
											('moving_mean' in var.name and 'Momentum' not in var.name) or
											('moving_variance' in var.name and 'Momentum' not in var.name)]

		return fedmodel.ModelArchitecture(loss=fed_loss_tensor,
										  train_step=fed_train_op,
										  predictions=fed_predictions,
										  model_federated_variables=federated_variables_collection)