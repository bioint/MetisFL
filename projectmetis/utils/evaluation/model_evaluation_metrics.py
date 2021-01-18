import json
import scipy.stats
import sklearn.metrics

import pandas as pd
import numpy as np

# Set global numpy warning suppressors
np.seterr(divide='ignore', invalid='ignore')


class EvaluationMetrics(object):


	def __init__(self, is_train=False, is_validation=False, is_test=False, learner_id=None, ):
		self.is_train = is_train
		self.is_validation = is_validation
		self.is_test = is_test
		self.learner_id = learner_id


class Classification(EvaluationMetrics):

	def __init__(self, num_classes, negative_indices, target_stat_name='accuracy',
				 is_train=False, is_validation=False, is_test=False, learner_id=None):

		super().__init__(is_train, is_validation, is_test, learner_id)

		self.num_classes = num_classes
		self.negative_indices = negative_indices
		self.positive_indices = [i for i in range(self.num_classes) if i not in self.negative_indices]

		self.label_metrics = ['TP', 'TN', 'FP', 'FN', 'NoExamples']
		self.label_metrics_df = pd.DataFrame(data=np.zeros([num_classes, len(self.label_metrics)]),
											 index=range(0, num_classes),
											 columns=self.label_metrics)

		self.cumulative_metrics = ['Accuracy',
								   'Accuracy_ExcludeNegatives',
								   'Precision_ExcludeNegatives',
								   'Recall_ExcludeNegatives',
								   'F1_ExcludeNegatives']

		self.target_original_stat_name = target_stat_name.lower()
		if len(self.negative_indices) > 0:
			target_stat_name = target_stat_name.lower() + '_excludenegatives'
			self.session_target_stat_name = [x for x in self.cumulative_metrics if target_stat_name == x.lower()][0]
			self.eval_with_negative_indices = True
		else:
			target_stat_name = target_stat_name.lower()
			self.session_target_stat_name = [x for x in self.cumulative_metrics if target_stat_name == x.lower()][0]
			self.eval_with_negative_indices = False

		self.cumulative_metrics_df = pd.DataFrame(data=np.zeros([1, len(self.cumulative_metrics)]),
												  columns=self.cumulative_metrics)


	def retrieve_classification_metrics(self, confusion_mtx, logging=True, to_json=True,
										include_json_confusion_matrix=False,
										include_json_evaluation_per_class=False):
		"""

		Args:
			confusion_mtx: matrix of size n x n, with rows being the actual labels and columns the prediction labels
			logging: whether to print to the console or not
			to_json
			include_json_confusion_matrix:
			include_json_evaluation_per_class:
		Returns:

		"""
		for cid in range(self.num_classes):

			# Class True Positives
			cid_tp = confusion_mtx[cid, cid]

			# Class False Positives
			cid_fp = np.sum(confusion_mtx[:, cid]) - confusion_mtx[cid, cid]

			# Class False Negatives
			cid_fn = np.sum(confusion_mtx[cid, :]) - confusion_mtx[cid, cid]

			# Class True Negatives
			cid_tn = np.sum(confusion_mtx) - cid_tp - cid_fp - cid_fn

			# Class Number of examples
			cid_no_examples = cid_tp + cid_fn

			res = (cid_tp, cid_tn, cid_fp, cid_fn, cid_no_examples)

			for idx, col in enumerate(self.label_metrics_df.columns):
				new_val = self.label_metrics_df.iloc[cid][col] + res[idx]
				self.label_metrics_df.at[cid, col] = new_val

		# Cumulative(ctv) Computations
		ctps = np.sum(self.label_metrics_df['TP'])
		ctps_wout_neg = ctps - np.sum(self.label_metrics_df.iloc[self.negative_indices]['TP'])
		cfps = np.sum(self.label_metrics_df['FP'])
		cfps_wout_neg = cfps - np.sum(self.label_metrics_df.iloc[self.negative_indices]['FP'])
		cfns = np.sum(self.label_metrics_df['FN'])
		cfns_wout_neg = cfns - np.sum(self.label_metrics_df.iloc[self.negative_indices]['FN'])
		ctns = np.sum(self.label_metrics_df['TN'])
		total_examples = np.sum(self.label_metrics_df['NoExamples'])

		# Cumulative predictions is the same as total Number of Examples
		ctv_predictions = np.sum(confusion_mtx)
		ctv_predictions_wout_neg = ctv_predictions \
										- np.sum(confusion_mtx[self.negative_indices, :]) \
										- np.sum(confusion_mtx[:, self.negative_indices]) \
										+ np.sum(confusion_mtx[self.negative_indices, self.negative_indices])

		# The following is similar to saying #TP/#(Number_of_Examples)
		ctv_accuracy = np.divide(ctps, ctv_predictions, out=np.array(0.0), where=ctv_predictions!=0)
		ctv_accuracy_wout_neg = np.divide(ctps_wout_neg, ctv_predictions_wout_neg, out=np.array(0.0),
										  where=ctv_predictions_wout_neg!=0)
		# Cumulative Precision across all classes
		micro_precision = np.divide(ctps, ctps + cfps, out=np.array(0.0), where=(ctps+cfps)!=0)
		micro_precision_wout_neg = np.divide(ctps_wout_neg, ctps_wout_neg + cfps_wout_neg, out=np.array(0.0),
											 where=(ctps_wout_neg + cfps_wout_neg)!=0)
		# Cumulative Recall across all classes
		micro_recall = np.divide(ctps, ctps + cfns, out=np.array(0.0), where=(ctps + cfns)!=0)
		micro_recall_wout_neg = np.divide(ctps_wout_neg, ctps_wout_neg + cfns_wout_neg, out=np.array(0.0),
										  where=(ctps_wout_neg + cfns_wout_neg)!=0)
		# Cumulative F1-Score across all classes
		micro_f1_score = np.divide(2 * micro_precision * micro_recall, micro_precision + micro_recall,
								   out=np.array(0.0), where=(micro_precision + micro_recall)!=0)
		micro_f1_score_wout_neg = np.divide(2 * micro_precision_wout_neg * micro_recall_wout_neg,
											micro_precision_wout_neg + micro_recall_wout_neg, out=np.array(0.0),
											where=(micro_precision_wout_neg + micro_recall_wout_neg)!=0)
		res = (ctv_accuracy, ctv_accuracy_wout_neg, micro_precision, micro_precision_wout_neg, micro_recall,
			   micro_recall_wout_neg, micro_f1_score, micro_f1_score_wout_neg)
		for idx, col in enumerate(self.cumulative_metrics_df.columns):
			new_val = self.cumulative_metrics_df.iloc[0][col] + res[idx]
			self.cumulative_metrics_df.at[0, col] = new_val

		# Replace all NaN Values with 0s
		self.cumulative_metrics_df.fillna(0)
		dataset_type = 'Train' if self.is_train else 'Validation' if self.is_validation else 'Test'
		if logging:
			if self.eval_with_negative_indices:
				print("{}, {}\n\t{}".format(
					self.learner_id,
					dataset_type,
					self.cumulative_metrics_df.loc[:, self.cumulative_metrics_df.columns.isin(
						['Accuracy_ExcludeNegatives', 'Precision_ExcludeNegatives', 'Recall_ExcludeNegatives',
						 'F1_ExcludeNegatives'])]))
			else:
				print("{}, {}\n\t{}".format(
					self.learner_id,
					dataset_type,
					self.cumulative_metrics_df.loc[:, self.cumulative_metrics_df.columns.isin(['Accuracy'])]))

		target_session_stat_score = self.cumulative_metrics_df.at[0, self.session_target_stat_name]

		if to_json:
			json_obj = { self.target_original_stat_name: target_session_stat_score }
			if include_json_confusion_matrix:
				json_obj["confusion_matrix"] = confusion_mtx.tolist()
			if include_json_evaluation_per_class:
				json_obj["evaluation_per_class"] = json.loads(
					self.label_metrics_df[['TP', 'TN', 'FP', 'FN', 'NoExamples']].to_json(orient='index'))
			return json_obj
		else:
			return target_session_stat_score, self.label_metrics_df, self.cumulative_metrics_df


class Regression(EvaluationMetrics):

	def __init__(self, is_train=False, is_validation=False, is_test=False, learner_id=None):
		super().__init__(is_train, is_validation, is_test, learner_id)

		self.num_examples = np.nan
		self.mae = np.nan
		self.mse = np.nan
		self.rmse = np.nan
		self.corr = np.nan
		self.rcorr = np.nan


	def retrieve_regression_metrics(self, ground_truth_values, predicted_values):

		# TODO HACK for DVW on ADNI dataset!!!
		#  We need to filter out the ground truth values which are >=55
		#  Find their index and then remove them from both ground truth and predicted collections
		# if self.is_validation:
		# 	predicted_values = predicted_values[np.where(ground_truth_values >= 55)]
		# 	ground_truth_values = ground_truth_values[np.where(ground_truth_values >= 55)]

		if not isinstance(ground_truth_values, np.ndarray):
			ground_truth_values = np.array(ground_truth_values)
		if not isinstance(predicted_values, np.ndarray):
			predicted_values = np.array(predicted_values)

		# Check if both input matrices have at least one element.
		not_empty_inputs = ground_truth_values.size != 0 and predicted_values.size != 0
		# Check if both input matrices do not have elements with infinite values.
		not_infinite_inputs = not np.any(np.isinf(ground_truth_values)) and not np.any(np.isinf(predicted_values))
		# Check if both input matrices do not have elements with NaN values.
		not_nan_inputs = not np.any(np.isnan(ground_truth_values)) and not np.any(np.isnan(predicted_values))

		# Make sure that both ground truth and predicted matrices are not empty and not infinite.
		if not_empty_inputs and not_infinite_inputs and not_nan_inputs:
			self.num_examples = int(ground_truth_values.size)
			self.mae = sklearn.metrics.mean_absolute_error(ground_truth_values, predicted_values)
			self.mse = sklearn.metrics.mean_squared_error(ground_truth_values, predicted_values)
			self.rmse = self.mse ** 0.5
			# We know that in some cases the 'stats.pearsonr' function will return NaN so we suppress these warnings.
			with np.errstate(divide='ignore', invalid='ignore'):
				self.corr = scipy.stats.pearsonr(ground_truth_values, predicted_values)[0]
				self.rcorr = scipy.stats.pearsonr(ground_truth_values, ground_truth_values - predicted_values)[0]

		# We explicitly cast the values to python floats, in order to avoid numpy data types json serialization errors
		# when dumping the final results to the json execution file.
		json_obj = {'num_examples': self.num_examples, 'mse': float(self.mse), 'rmse': float(self.rmse),
					'mae': float(self.mae), 'corr': float(self.corr), 'rcorr': float(self.rcorr)}

		return json_obj
