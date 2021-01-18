import federation.fed_model as fedmodel
import tensorflow as tf

from utils.brainage import loss_fun
from utils.tf import tf_ops_fedprox


class BrainAgeCNNFedModel(fedmodel.FedModelDef):
	"""
	A Simple Fully Connected Convolution Neural Network for Brain Age prediction.
	"""

	def __init__(self, distribution_based_training=True, distribution_type="kld"):
		self.distribution_based_training = distribution_based_training
		self.distribution_type = distribution_type


	def input_tensors_datatype(self, **kwargs):
		# return {'images': tf.placeholder(tf.float32, [None, 182, 218, 182, 1], name='images')}
		return {'images': tf.placeholder(tf.float32, [None, 91, 109, 91, 1], name='images')}


	def output_tensors_datatype(self, **kwargs):
		return {'dist': tf.placeholder(tf.float32, [None, 36], name='dist'),
				'age': tf.placeholder(tf.float32, [None, 1], name='age')}


	def model_architecture(self, input_tensors, output_tensors,
						   global_step=None, batch_size=None, dataset_size=None, **kwargs):

		x_images = input_tensors['images']
		y_output = output_tensors['age']

		# This is the dropout placeholder. It is set to true only when performing a training step.
		is_training = tf.placeholder(tf.bool, name='is_training')

		def get_ages_from_distribution(input_distribution):
			# Bins for prob distributions
			bins = tf.range(45, limit=81, delta=1.0)
			bins = tf.expand_dims(bins, axis=0)

			# Calculate expected value (Age pred)
			ages = tf.matmul(bins, input_distribution, transpose_b=True)
			ages = tf.expand_dims(ages, axis=1)

			return ages


		def calculate_distribution_loss(loss_metric, true_dists, pred_dists):
			"""
			A helper function to select and compute the
			total loss given the prediction and true values.
			:param loss_metric:
			:param pred_dists:
			:param true_dists:
			:return:
			"""

			# Build loss portion of Graph
			if loss_metric == "kld":
				loss_fun.kld_loss(pred_dists, true_dists, collection_name="losses")

			elif loss_metric == "rkld":
				loss_fun.rkld_loss(pred_dists, true_dists, collection_name="losses")

			elif loss_metric == "skld":
				loss_fun.skld_loss(pred_dists, true_dists, collection_name="losses")

			elif loss_metric == "js":
				loss_fun.js_loss(pred_dists, true_dists, collection_name="losses")

			elif loss_metric == "bhat":
				loss_fun.bhat_loss(pred_dists, true_dists, collection_name="losses")

			elif loss_metric == "hell":
				loss_fun.hell_loss(pred_dists, true_dists, collection_name="losses")

			# Assemble all of the losses for the current tower
			losses = tf.get_collection("losses")

			# Calculate the total loss for the current tower
			total_loss = tf.add_n(losses, name="total_loss")

			return total_loss


		def calculate_mse_loss(ground_truths, predictions):

			# By default the collection name of mse loss is tf.GraphKeys.LOSSES
			tf.losses.mean_squared_error(labels=ground_truths, predictions=predictions)
			losses = tf.get_collection(tf.GraphKeys.LOSSES)
			total_loss = tf.add_n(losses, name="total_loss")

			return total_loss


		def compute_loss(ground_truths, predictions, distribution_based_training=True, distribution_type="kld"):

			if distribution_based_training:
				total_loss = calculate_distribution_loss(distribution_type, ground_truths, predictions)
			else:
				total_loss = calculate_mse_loss(ground_truths, predictions)

			return total_loss


		def sgd_optimizer(opt, learning_rate, momentum, global_step, decay_rate=0.0):
			"""
			A helper function to select the network optimizer
			:param opt:
			:param learning_rate:
			:param decay_rate:
			:param momentum:
			:param global_step:
			:return:
			"""

			# When decay rate is 0.0, then the learning rate remains invariable/immutable. TF executes the following:
			# 	decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
			learning_rate = tf.train.inverse_time_decay(
				learning_rate=learning_rate, global_step=global_step, decay_steps=1.0,
				decay_rate=decay_rate)

			if opt == "adam":
				return tf.train.AdamOptimizer(learning_rate=learning_rate)
			elif opt == "momentum":
				return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
			elif opt == "fedprox":
				return tf_ops_fedprox.PerturbedOptimizerWithFedProx(learning_rate=learning_rate, mu=0.001)


		def convolution_block_v2(inputs, num_filters, name):

			with tf.variable_scope(name):
				inputs = tf.keras.layers.Conv3D(num_filters, 3, strides=1, padding="same", name=name + "_conv")(inputs)
				inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
				inputs = tf.keras.layers.MaxPooling3D(2, strides=2, padding="valid", name=name + "_max_pool")(inputs)
				inputs = tf.nn.relu(inputs, name=name + "_relu")


			return inputs


		def infer_ages_v2(images, is_training, distribution_based_training):

			with tf.variable_scope("Brain_Age_Model"):

				inputs = convolution_block_v2(images, 32, "conv_block1")
				inputs = convolution_block_v2(inputs, 64, "conv_block2")
				inputs = convolution_block_v2(inputs, 128, "conv_block3")
				inputs = convolution_block_v2(inputs, 256, "conv_block4")
				inputs = convolution_block_v2(inputs, 256, "conv_block5")

				# Last Layer
				inputs = tf.keras.layers.Conv3D(64, 1, strides=1, name="post_conv1")(inputs)
				inputs = tf.contrib.layers.instance_norm(inputs, center=False, scale=False)
				inputs = tf.nn.relu(inputs, name="post_relu")
				inputs = tf.keras.layers.AveragePooling3D(pool_size=(2, 3, 2), name="post_avg_pool")(inputs)

				# Default rate: 0.5
				drop = tf.layers.dropout(inputs, rate=0.5, training=is_training, name="drop")

				if distribution_based_training:
					# Output prob dist.
					output = tf.layers.dense(drop, 36, activation=tf.nn.softmax, name="predicted_distribution")
					output = tf.squeeze(output)
				else:
					# Regression (MSE).
					output = tf.layers.Conv3D(1, kernel_size=1, strides=1, name="reg_conv",
											  bias_initializer=tf.constant_initializer(62.68))(drop)
					# output = tf.layers.Conv3D(1, kernel_size=1, strides=1, name="reg_conv")(drop)

					output = tf.squeeze(output)

				return output


		def whole_batch_training_operation(current_loss, global_step):
			# Define optimizer
			# optimizer = sgd_optimizer("adam", 5e-5, 0.0, 0.0, global_step)
			optimizer = sgd_optimizer(opt="momentum", learning_rate=5e-5, momentum=0.0, global_step=global_step)
			# optimizer = sgd_optimizer(opt="fedprox", learning_rate=5e-5, momentum=0.0, global_step=global_step)

			# Gradient clipping.
			# grads_and_vars = optimizer.compute_gradients(current_loss, tf.trainable_variables())
			# grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=1.0), var) for grad, var in grads_and_vars]
			# train_op = optimizer.apply_gradients(grads_and_vars)

			# Normal Optimization Step
			train_op = optimizer.minimize(current_loss)

			return train_op


		model_inferred_ages = infer_ages_v2(images=x_images, is_training=is_training,
											distribution_based_training=self.distribution_based_training)
		total_loss = compute_loss(y_output, model_inferred_ages,
								  distribution_based_training=self.distribution_based_training,
								  distribution_type=self.distribution_type)
		train_op = whole_batch_training_operation(total_loss, global_step)

		if self.distribution_based_training:
			model_inferred_ages = get_ages_from_distribution(model_inferred_ages)

		# Federated Model Properties
		fed_loss_tensor = fedmodel.FedTensor(tensor=total_loss, feed_dict={is_training: False})
		fed_train_step_op = fedmodel.FedOperation(operation=train_op, feed_dict={is_training: True})
		fed_predictions = fedmodel.FedTensor(tensor=model_inferred_ages, feed_dict={is_training: False})
		fed_trainable_variables_collection = tf.trainable_variables(scope="Brain_Age_Model")


		return fedmodel.ModelArchitecture(loss=fed_loss_tensor,
										  train_step=fed_train_step_op,
										  predictions=fed_predictions,
										  model_federated_variables=fed_trainable_variables_collection)
