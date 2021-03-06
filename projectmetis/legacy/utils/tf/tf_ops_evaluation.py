import numpy as np
import tensorflow as tf

from collections import OrderedDict
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from utils.evaluation.model_evaluation_metrics import Classification, Regression


class TFGraphEvaluation(object):

	def __init__(self, x_placeholders, y_placeholders, predictions_tensor_fedmodel, loss_tensor_fedmodel=None,
				 is_classification=False, is_regression=False, training_init_op=None, next_train_dataset=None,
				 validation_init_op=None, next_validation_dataset=None, testing_init_op=None, next_test_dataset=None,
				 negative_classes_indices=None, num_classes=None, is_eval_output_scalar=None):

		assert (is_classification is True or is_regression is True), \
			"Need to specify whether this is a classification or regression evaluation task."

		self.x_placeholders = x_placeholders
		self.y_placeholders = y_placeholders
		self.predictions_tensor_fedmodel = predictions_tensor_fedmodel
		self.loss_tensor_fedmodel = loss_tensor_fedmodel
		self.is_classification = is_classification
		self.is_regression = is_regression

		# Evaluation datasets related operations.
		self.training_init_op = training_init_op
		self.next_train_dataset = next_train_dataset
		self.validation_init_op = validation_init_op
		self.next_validation_dataset = next_validation_dataset
		self.testing_init_op = testing_init_op
		self.next_test_dataset = next_test_dataset

		"""
		The following class members will be updated once we register the tensorflow evaluation operations.
			1. For classification the evaluation operation update the confusion matrix counters.
			2. For regression, we update the collections/accumulators of predictions and ground truth
		The eval_accumulator, returns the result of the LAST update. Namely:
			1. For classification the eval_accumulator returns the latest confusion matrix.
			2. For regression, the eval_accumulator returns the collections of [ground truth and predictions values].
		"""
		self.eval_accumulator_update_op = None
		self.eval_accumulator = None
		self.prediction_terms_reset_op = None

		if self.is_classification:
			if num_classes is None and num_classes is None and is_eval_output_scalar is None:
				raise RuntimeError("For classification tasks, we need to specify the number of classes, "
								   "the negative classes (if any) and whether the evaluation output is a scalar.")
			else:
				self.is_eval_output_scalar = is_eval_output_scalar
				self.num_classes = num_classes
				self.negative_classes_indices = negative_classes_indices


	def assign_tfdatasets_operators(self, training_init_op=None, next_train_dataset=None, validation_init_op=None,
									next_validation_dataset=None, testing_init_op=None, next_test_dataset=None):
		self.training_init_op = training_init_op
		self.next_train_dataset = next_train_dataset
		self.validation_init_op = validation_init_op
		self.next_validation_dataset = next_validation_dataset
		self.testing_init_op = testing_init_op
		self.next_test_dataset = next_test_dataset


	def register_evaluation_ops(self, y_eval_placeholder_name, batch_size=None):

		with tf.variable_scope('model_prediction_terms'):

			if self.is_classification:

				if self.is_eval_output_scalar:
					assert batch_size is not None, \
						"We need to specify the batch size when our classification evaluation output are scalars."
					weights = tf.sequence_mask(np.ones(shape=[batch_size]))
				else:
					if 'sequence_lengths' in self.x_placeholders:
						weights = tf.sequence_mask(self.x_placeholders['sequence_lengths'])
					else:
						raise ValueError("'sequence_lengths' is not provided in the input")

				conf_matrix, conf_matrix_op = _streaming_confusion_matrix(
					labels=self.y_placeholders[y_eval_placeholder_name],
					predictions=self.predictions_tensor_fedmodel.get_tensor(),
					num_classes=self.num_classes, weights=weights)

				self.eval_accumulator_update_op = conf_matrix_op
				self.eval_accumulator = conf_matrix
				stream_vars = [i for i in tf.local_variables() if i.name.split('/')[0] == 'model_prediction_terms']
				self.prediction_terms_reset_op = [tf.initializers.variables(stream_vars)]

			elif self.is_regression:
				ground_truth = tf.squeeze(self.y_placeholders[y_eval_placeholder_name])
				# Necessary condition when calling squeeze operation, because in the presence of a single element, it
				# will return a single element. E.g. tf.squeeze([[[68.31521]]]) -> array(68.31521) NOT array([68.31521]).
				ground_truth = tf.reshape(ground_truth, shape=[tf.size(ground_truth), ])
				prediction = self.predictions_tensor_fedmodel.get_tensor()
				prediction = tf.squeeze(prediction)
				prediction = tf.reshape(prediction, shape=[tf.size(prediction), ])

				with tf.variable_scope('model_prediction_terms'):
					# Defining a collection for the accumulators will raise a Variable uninitialized error!
					ground_truths_var = tf.Variable([np.inf], trainable=False, validate_shape=False, dtype=tf.float32,
													name="GroundTruthAccumulator")
					predictions_var = tf.Variable([np.inf], trainable=False, validate_shape=False, dtype=tf.float32,
												  name="PredictionsAccumulator")
					self.prediction_terms_reset_op = [ground_truths_var.initializer, predictions_var.initializer]

				# Conditional Assignment. If it is the first time we assign a value to the variable then we assign the
				# batch as it is. In essence, we want to remove the np.infinity value from the variable, thus we check
				# whether its first element is equal to np.inf.
				ground_truths_accumulator_update_op = tf.cond(tf.equal(ground_truths_var[0], np.inf),
															  true_fn=lambda: tf.assign(ground_truths_var, ground_truth,
																						validate_shape=False),
															  false_fn=lambda: tf.assign(ground_truths_var, tf.concat(
																  [ground_truths_var, ground_truth], axis=0),
																						 validate_shape=False))
				predictions_accumulator_update_op = tf.cond(tf.equal(predictions_var[0], np.inf),
															true_fn=lambda: tf.assign(predictions_var, prediction,
																					  validate_shape=False),
															false_fn=lambda: tf.assign(predictions_var, tf.concat(
																[predictions_var, prediction], axis=0),
																					   validate_shape=False))
				# Make sure we do not feed/concat empty values
				ground_truths_accumulator_update_op = tf.cond(
					tf.logical_and(
						tf.not_equal(tf.size(ground_truth), 0), tf.not_equal(tf.size(prediction), 0)),
					true_fn=lambda: ground_truths_accumulator_update_op,
					false_fn=lambda: ground_truths_var)

				predictions_accumulator_update_op = tf.cond(
					tf.logical_and(
						tf.not_equal(tf.size(ground_truth), 0), tf.not_equal(tf.size(prediction), 0)),
					true_fn=lambda: predictions_accumulator_update_op,
					false_fn=lambda: predictions_var)

				self.eval_accumulator_update_op = [ground_truths_accumulator_update_op, predictions_accumulator_update_op]
				self.eval_accumulator = [ground_truths_var, predictions_var]


	def evaluate_model_on_existing_graph(self,
										 tf_session,
										 target_stat_name,
										 learner_id=None,
										 include_json_evaluation_per_class=False,
										 include_json_confusion_matrix=False,
										 compute_losses=False,
										 eval_train=True,
										 eval_valid=True,
										 eval_test=True):

		# We use np.nansum, np.nanmean and np.nanvar to suppress the division warnings.
		np_sum = lambda x: float(np.nansum(x)) if x else np.nan
		np_squared_sum = lambda x: float(np.nansum(np.square(x))) if x else np.nan
		np_mean = lambda x: float(np.nanmean(x)) if x else np.nan
		np_var = lambda x: float(np.nanvar(x)) if x else np.nan
		train_eval_metrics = dict()
		valid_eval_metrics = dict()
		test_eval_metrics = dict()

		if eval_train:
			# Evaluate model on train dataset.
			train_eval_result, train_losses = self.__dataset_evaluation(tf_session, is_train=True,
																		compute_losses=compute_losses)
			train_eval_metrics = self.__dataset_evaluation_metrics(train_eval_result, target_stat_name,
									is_train=True, learner_id=learner_id,
									include_json_evaluation_per_class=include_json_evaluation_per_class,
									include_json_confusion_matrix=include_json_confusion_matrix)
			train_eval_metrics['train_loss_sum'] = np_sum(train_losses)
			train_eval_metrics['train_loss_squared_sum'] = np_squared_sum(train_losses)
			train_eval_metrics['train_loss_mean'] = np_mean(train_losses)
			train_eval_metrics['train_loss_variance'] = np_var(train_losses)

		if eval_valid:
			# Evaluate model on validation dataset.
			valid_eval_result, valid_losses = self.__dataset_evaluation(tf_session, is_validation=True,
																		compute_losses=compute_losses)
			valid_eval_metrics = self.__dataset_evaluation_metrics(valid_eval_result, target_stat_name,
									is_validation=True, learner_id=learner_id,
									include_json_evaluation_per_class=include_json_evaluation_per_class,
									include_json_confusion_matrix=include_json_confusion_matrix)
			valid_eval_metrics['validation_loss_sum'] = np_sum(valid_losses)
			valid_eval_metrics['validation_loss_squared_sum'] = np_squared_sum(valid_losses)
			valid_eval_metrics['validation_loss_mean'] =  np_mean(valid_losses)
			valid_eval_metrics['validation_loss_variance'] = np_var(valid_losses)

		if eval_test:
			# Evaluate model on test dataset.
			test_eval_result, test_losses = self.__dataset_evaluation(tf_session, is_test=True,
																	  compute_losses=compute_losses)
			test_eval_metrics = self.__dataset_evaluation_metrics(test_eval_result, target_stat_name,
									is_test=True, learner_id=learner_id,
									include_json_evaluation_per_class=include_json_evaluation_per_class,
									include_json_confusion_matrix=include_json_confusion_matrix)
			test_eval_metrics['test_loss_sum'] = np_sum(test_losses)
			test_eval_metrics['test_loss_squared_sum'] = np_squared_sum(test_losses)
			test_eval_metrics['test_loss_mean'] = np_mean(test_losses)
			test_eval_metrics['test_loss_variance'] = np_var(test_losses)

		return train_eval_metrics, valid_eval_metrics, test_eval_metrics


	def __dataset_evaluation(self, tf_session, is_train=False, is_validation=False, is_test=False,
							 compute_losses=False):

		assert any([is_train, is_validation, is_test]), "Need to specify at least one dataset type."

		if is_train:
			dataset_init_op = self.training_init_op
			dataset_next = self.next_train_dataset
		elif is_validation:
			dataset_init_op = self.validation_init_op
			dataset_next = self.next_validation_dataset
		elif is_test:
			dataset_init_op = self.testing_init_op
			dataset_next = self.next_test_dataset

		# Evaluate the model against assigned dataset. We need both the iterator and next dataset to be not None.
		dataset_losses = []
		if dataset_init_op is not None and dataset_next is not None:
			tf_session.run(dataset_init_op)
			tf_session.run(self.prediction_terms_reset_op)
			while True:

				try:
					# Initialize tf dataset iterator.
					batch = tf_session.run(dataset_next)

					# Build feed dictionary with input data.
					extra_feeds = OrderedDict()
					for placeholder_name, placeholder_def in self.x_placeholders.items():
						extra_feeds[placeholder_def] = batch[placeholder_name]
					for placeholder_name, placeholder_def in self.y_placeholders.items():
						extra_feeds[placeholder_def] = batch[placeholder_name]

					# We need to retrieve any additional placeholders declared during the construction of the network.
					# In this step we need to get the placeholders for the federated model inference operation.
					# 	e.g. {'is_training': False}
					extra_feeds.update(self.predictions_tensor_fedmodel.get_feed_dictionary())
					tf_session.run(self.eval_accumulator_update_op, feed_dict=extra_feeds)

					if compute_losses:
						# Compute loss of given batch.
						extra_feeds.update(self.loss_tensor_fedmodel.get_feed_dictionary())
						batch_loss = tf_session.run(self.loss_tensor_fedmodel.get_tensor(), feed_dict=extra_feeds)
						# Append the loss for each example. For instance a batch may be comprised of 5 examples, so we
						# need to add all the losses, thus the flattening operation and the list extension.
						batch_loss = batch_loss.flatten()
						dataset_losses.extend(batch_loss)

				except tf.errors.OutOfRangeError:
					break

		evaluation_result = tf_session.run(self.eval_accumulator)

		return evaluation_result, dataset_losses


	def __dataset_evaluation_metrics(self, eval_result, target_stat_name,
								   is_train=False, is_validation=False, is_test=False, learner_id=None,
								   include_json_evaluation_per_class=False,
								   include_json_confusion_matrix=False):

		assert any([is_train, is_validation, is_test]), "Need to specify at least one dataset type."

		if self.is_classification:

			classification_metrics = Classification(num_classes=self.num_classes,
													negative_indices=self.negative_classes_indices,
													target_stat_name=target_stat_name,
													is_train=is_train,
													is_validation=is_validation,
													is_test=is_test,
													learner_id=learner_id)

			# The following returns the test set evaluation of per class
			eval_metrics = classification_metrics.retrieve_classification_metrics(
				confusion_mtx=eval_result, include_json_evaluation_per_class=include_json_evaluation_per_class,
				include_json_confusion_matrix=include_json_confusion_matrix)
		else:
			regression_metrics = Regression(is_train, is_validation, is_test, learner_id)
			ground_truth = eval_result[0]
			prediction = eval_result[1]
			eval_metrics = regression_metrics.retrieve_regression_metrics(ground_truth, prediction)

		return eval_metrics
