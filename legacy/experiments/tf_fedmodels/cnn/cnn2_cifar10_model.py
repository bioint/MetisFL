import federation.fed_model as fedmodel
import tensorflow as tf

from utils.tf.tf_ops_fedprox import PerturbedOptimizerWithFedProx

class Cifar10FedModel(fedmodel.FedModelDef):

	def __init__(self, learning_rate=0.1, momentum=0.0, run_with_distorted_images=True):
		# CONSTANTS DESCRIBING THE TRAINING PROCESS
		self.moving_average_decay = 0.9999  # The decay to use for the moving average.
		# self.num_epochs_per_decay = 350.0  # Epochs after which learning rate decays.
		# self.num_epochs_per_decay = 5.0  # Epochs after which learning rate decays.
		# self.learning_rate_decay_factor = 0.01 # Learning rate decay factor.
		# self.learning_rate_decay_factor = 0.992 # 0.992  # Learning rate decay factor.
		self.initial_learning_rate = learning_rate  # Initial learning rate.
		self.sgd_momentum = momentum

		# IMAGE DATASET SPECIFICS
		self.run_with_distorted_images = run_with_distorted_images
		self.img_depth = 3
		if self.run_with_distorted_images:
			self.img_height = 24
			self.img_width = 24
			self.weight_input_dimension_local3_layer = 2304
		else:
			self.img_height = 32
			self.img_width = 32
			self.weight_input_dimension_local3_layer = 4096


	def input_tensors_datatype(self):
		# image: 3-D Tensor of [height, width, 3] of type.float32
		# img_original_size = (24,24,3)
		# img_flatten_size = 1728
		return { 'images': tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_depth]) }


	def output_tensors_datatype(self):
		return { 'labels': tf.placeholder(tf.int64, [None]) }


	def model_architecture(self, input_tensors, output_tensors, global_step, batch_size, dataset_size, **kwargs):

		"""

		:param input_tensors: the input data to the network
		:param output_tensors: output data of the network
		:param global_step
		:param batch_size
		:param dataset_size

		:return:
		"""

		input = input_tensors['images']
		output = output_tensors['labels']

		learner_training_devices = kwargs['learner_training_devices'] if 'learner_training_devices' in kwargs else None
		weight_input_dimension_local3_layer = self.weight_input_dimension_local3_layer

		def inference(images):
			"""Build the CIFAR-10 model.
			Args:
			  images: Images returned from distorted_inputs() or inputs().
			Returns:
			  Logits.
			"""

			var_index = 0
			# We instantiate all variables using tf.get_variable() instead of
			# tf.Variable() in order to share variables across multiple GPU training runs.
			# If we only ran this model on a single GPU, we could simplify this function
			# by replacing all instances of tf.get_variable() with tf.Variable().

			# conv1
			with tf.variable_scope('conv1') as scope:
				W_conv1 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2, dtype=tf.float32), name='weight_conv1')
				conv = tf.nn.conv2d(images, W_conv1, [1, 1, 1, 1], padding='SAME')
				b_conv1 = tf.Variable(initial_value=tf.constant(value=0.0, shape=[64]), name='bias_conv1')
				pre_activation = tf.nn.bias_add(conv, b_conv1)
				conv1 = tf.nn.relu(pre_activation, name=scope.name)
				var_index += 2

			# pool1
			pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
			# norm1
			norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

			# conv2
			with tf.variable_scope('conv2') as scope:
				W_conv2 = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2, dtype=tf.float32), name='weight_conv2')
				conv = tf.nn.conv2d(norm1, W_conv2, [1, 1, 1, 1], padding='SAME')
				b_conv2 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[64]), name='bias_conv2')
				pre_activation = tf.nn.bias_add(conv, b_conv2)
				conv2 = tf.nn.relu(pre_activation, name=scope.name)
				var_index += 2

			# norm2
			norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

			# pool2
			pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

			# local3
			with tf.variable_scope('local3') as scope:
				# Move everything into depth so we can perform a single matrix multiply.
				 # 2304 for 24x24 images , 4096 for 32x32 images
				reshape = tf.reshape(pool2, [-1, weight_input_dimension_local3_layer])
				W_local3 = tf.Variable(initial_value=tf.truncated_normal(shape=[weight_input_dimension_local3_layer, 384], stddev=0.04, dtype=tf.float32), name='weight_local3')
				b_local3 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[384]), name='bias_local3')
				local3 = tf.nn.relu(tf.matmul(reshape, W_local3) + b_local3, name=scope.name)
				var_index += 2

			# local4
			with tf.variable_scope('local4') as scope:
				W_local4 = tf.Variable(initial_value=tf.truncated_normal(shape=[384, 192], stddev=0.04, dtype=tf.float32), name='weight_local4')
				b_local4 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[192]), name='bias_local4')
				local4 = tf.nn.relu(tf.matmul(local3, W_local4) + b_local4, name=scope.name)
				var_index += 2

			# linear layer(WX + b),
			# We don't apply softmax here because
			# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
			# and performs the softmax internally for efficiency.
			with tf.variable_scope('softmax_linear') as scope:
				W_softmax_linear = tf.Variable(initial_value=tf.truncated_normal(shape=[192, 10], stddev=1 / 192.0, dtype=tf.float32), name='weight_softmax_linear')
				b_softmax_linear = tf.Variable(initial_value=tf.constant(value=0.0, shape=[10]), name='bias_softmax_linear')
				softmax_linear = tf.add(tf.matmul(local4, W_softmax_linear), b_softmax_linear, name=scope.name)

			return softmax_linear


		def loss(logits, labels):
			"""Add L2Loss to all the trainable variables.
			Add summary for "Loss" and "Loss/avg".
			Args:
			  logits: Logits from inference().
			  labels: Labels from distorted_inputs or inputs(). 1-D tensor
					  of shape [batch_size]
			Returns:
			  Loss tensor of type float.
			"""
			# Calculate the average cross entropy loss across the batch.
			labels = tf.cast(labels, tf.int64)
			cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
			cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
			tf.add_to_collection('losses', cross_entropy_mean)

			# The total loss is defined as the cross entropy loss plus all of the weight
			# decay terms (L2 loss).
			return tf.add_n(tf.get_collection('losses'), name='total_loss_tensor')


		def _add_losses(total_loss):
			"""Add summaries for losses in CIFAR-10 model.
			Generates moving average for all losses and associated summaries for
			visualizing the performance of the network.
			Args:
			  total_loss: Total loss from loss().
			Returns:
			  loss_averages_op: op for generating moving averages of losses.
			"""
			# Compute the moving average of all individual losses and the total loss.
			loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
			losses = tf.get_collection('losses')
			loss_averages_op = loss_averages.apply(losses + [total_loss])

			return loss_averages_op


		def sgd_optimizer():
			""" VANILLA SGD """
			# opt = tf.train.GradientDescentOptimizer(initial_learning_rate=self.initial_learning_rate)

			# """ VANILLA SGD With CLR [min-max]: [lr/2 - lr] """
			# opt = tf.train.GradientDescentOptimizer(initial_learning_rate=cyclic_learning_rate(global_step=global_step, initial_learning_rate=self.initial_learning_rate/4, max_lr=self.initial_learning_rate, step_size_half_cycle=step_size/2))

			""" Momentum SGD """
			opt = tf.train.MomentumOptimizer(learning_rate=self.initial_learning_rate, momentum=self.sgd_momentum)
			# opt = PerturbedOptimizerWithFedProx(learning_rate=self.initial_learning_rate, mu=0.001)


			""" Momentum SGD With Fed LR """
			# lr_annealing_value = tf.placeholder(tf.float32, name="lr_annealing_value")
			# momentum_annealing_value = tf.placeholder(tf.float32, name="momentum_annealing_value")
			#
			# lr_value = tf.multiply(self.initial_learning_rate, 1-lr_annealing_value, name="lr_value")
			# momentum_value = tf.multiply(self.sgd_momentum, 1+momentum_annealing_value, name="momentum_value")
			#
			# opt = tf.train.MomentumOptimizer(learning_rate=lr_value, momentum=momentum_value)

			# opt = tf.train.MomentumOptimizer(initial_learning_rate=self.initial_learning_rate, sgd_momentum=self.sgd_momentum)

			""" Momentum SGD With Triangular Epoch LR """
			# min_lr_value = tf.placeholder(tf.float32, name="min_lr")
			# max_lr_value = tf.placeholder(tf.float32, name="max_lr")
			# cycle_length = tf.placeholder(tf.float32, name="cycle_length")
			# current_cycle_step = tf.placeholder(tf.float32, name="current_cycle_step")

			# Step-Wise LR Increment
			# lr_step_value = tf.divide(tf.subtract(max_lr_value, min_lr_value), cycle_length)
			# lr_value = tf.add(min_lr_value, tf.multiply(current_cycle_step, lr_step_value),
			# 				  name="lr_value")

			# Triangular LR
			# half_cycle_length = tf.floor(tf.divide(cycle_length, 2))
			# lr_step_value = tf.divide(tf.subtract(max_lr_value, min_lr_value), half_cycle_length)
			# lhs_cycle_lr_value = tf.add(min_lr_value, tf.multiply(current_cycle_step, lr_step_value),
			# 							name="lhs_cycle_lr_value")
			# rhs_cycle_lr_value = tf.subtract(max_lr_value, tf.multiply(tf.subtract(current_cycle_step, half_cycle_length), lr_step_value),
			# 								 name="rhs_cycle_lr_value")
			# lr_value = tf.cond(pred=current_cycle_step < half_cycle_length,
			# 				   true_fn=lambda: lhs_cycle_lr_value,
			# 				   false_fn=lambda: rhs_cycle_lr_value)
			# opt = tf.train.MomentumOptimizer(initial_learning_rate=lr_value, sgd_momentum=self.sgd_momentum)

			""" Momentum SGD with Annealing """
			# global_update_scalar_clock = tf.placeholder(tf.float32, name="global_update_scalar_clock")
			# learner_global_update_scalar_clock = tf.placeholder(tf.float32, name="learner_global_update_scalar_clock")
			# lr_annealing = tf.divide(self.initial_learning_rate, tf.sqrt(
			# 	tf.add(
			# 		tf.subtract(global_update_scalar_clock, learner_global_update_scalar_clock),
			# 		1
			# 	)), name="AnnealingLR")
			# lr_annealing = tf.divide(self.initial_learning_rate, tf.sqrt(
			# 	tf.add(global_update_scalar_clock, 1
			# )), name="AnnealingLR")
			# opt = tf.train.MomentumOptimizer(initial_learning_rate=lr_annealing, sgd_momentum=self.sgd_momentum)

			""" Momentum SGD With Exponential Decay: Reduce by lr * exp(#UR/#LE) """
			# learner_update_requests = tf.placeholder(tf.float32, name="learner_update_requests")
			# learner_completed_epochs = tf.placeholder(tf.float32, name="learner_completed_epochs")
			# exp_lr = tf.multiply(self.initial_learning_rate,
			# 					 tf.exp(
			# 						 tf.negative(
			# 							 tf.divide(
			# 								 learner_update_requests,
			# 								 learner_completed_epochs))), name="ExpLR")
			# opt = tf.train.MomentumOptimizer(initial_learning_rate=exp_lr, sgd_momentum=self.sgd_momentum)

			""" Momentum SGD With CLR """
			# steps_per_epoch = tf.placeholder(tf.int32, name='steps_per_epoch')
			# min_lr = 0.01
			# max_lr = 0.1
			# cyclical_lr = cyclic_learning_rate(global_step=global_step,
			# 								   initial_learning_rate=min_lr,
			# 								   max_lr=max_lr,
			# 								   step_size_half_cycle=tf.divide(steps_per_epoch, 2),
			# 								   name='CyclicalLR')
			# opt = tf.train.MomentumOptimizer(initial_learning_rate=cyclical_lr, sgd_momentum=self.sgd_momentum)

			""" Momentum SGD With Cosine Decay - SGDR """
			# opt = tf.train.MomentumOptimizer(
			# 	initial_learning_rate=tf.train.cosine_decay_restarts(initial_learning_rate=self.initial_learning_rate, global_step=global_step, first_decay_steps=step_size),
			# 	sgd_momentum=self.sgd_momentum)

			""" ADAM OPT """
			# opt = tf.train.AdamOptimizer(initial_learning_rate=self.initial_learning_rate)

			return opt


		def train(total_loss):
			"""Train CIFAR-10 model.
			Create an optimizer and apply to all trainable variables. Add moving
			average for all trainable variables.
			Args:
			  total_loss: Total loss from loss().
			Returns:
			  train_op: op for training.
			"""

			# Generate moving averages of all losses and associated summaries.
			loss_averages_op = _add_losses(total_loss)

			# Compute gradients.
			with tf.control_dependencies([loss_averages_op]):
				opt = sgd_optimizer()
				grads = opt.compute_gradients(total_loss)

			# Apply gradients.
			apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

			# Track the moving averages of all trainable variables.
			variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
			with tf.control_dependencies([apply_gradient_op]):
				variables_averages_op = variable_averages.apply(tf.trainable_variables())
			return variables_averages_op


		def predictions(logits):
			return tf.argmax(logits, 1)


		def accuracy_value(logits):
			with tf.name_scope('accuracy_tensor') as scope:
				labels = tf.cast(output, tf.int64)
				top_k = tf.nn.in_top_k(logits, labels, 1)
				top_k_cast = tf.cast(top_k, dtype=tf.float32)
				accuracy = tf.reduce_mean(top_k_cast)
			return accuracy


		logits = inference(input)
		predictions_tensor = predictions(logits)
		total_loss_tensor = loss(logits, output)
		train_op = train(total_loss_tensor)
		accuracy_tensor = accuracy_value(logits)


		fed_loss_tensor = fedmodel.FedTensor(tensor=total_loss_tensor, feed_dict={})
		fed_train_op = fedmodel.FedOperation(operation=train_op, feed_dict={})
		fed_predictions = fedmodel.FedTensor(tensor=predictions_tensor, feed_dict={})
		fed_accuracy_tensor = fedmodel.FedTensor(tensor=accuracy_tensor, feed_dict={})
		trainable_variables_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

		# Add Momentum variables in the list of federated variables
		# sgd_momentum_variables_collection = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Momentum' in var.name]
		# trainable_variables_collection.extend(sgd_momentum_variables_collection)
		federated_variables_collection = trainable_variables_collection

		# User Defined Testing Variables
		moving_average_restoration_obj_name = "RestoringMovingAverages"
		variable_exponential_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step, name=moving_average_restoration_obj_name)
		model_testing_variables = variable_exponential_averages

		moving_average_obj_name = "ExponentialMovingAverage"
		fed_user_defined_variables = [tf_variable for var_dict_key, tf_variable in model_testing_variables.variables_to_restore().items() if moving_average_obj_name in var_dict_key]
		fed_user_defined_variables = fed_user_defined_variables


		return fedmodel.ModelArchitecture(train_step=fed_train_op,
										  predictions=fed_predictions,
										  accuracy=fed_accuracy_tensor,
										  model_federated_variables=federated_variables_collection,
										  user_defined_testing_variables_collection=fed_user_defined_variables,
										  loss=fed_loss_tensor)