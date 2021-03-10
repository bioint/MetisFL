import collections
import copy
import itertools
import json
import matplotlib
import re

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from scipy import stats
from utils.generic.time_ops import TimeUtil


class LogsProcessingUtil(object):

	FilenameLabels = collections.namedtuple(typename='FilenameLabels', field_names=['clients', 'community_function', 'learning_rate', 'sgd_momentum',
																					'batchsize', 'ufrequency', 'holdout', 'burnin', 'min_vloss',
																					'max_vloss', 'delta_le', 'is_adaptive'])
	CentralizedConfigs = collections.namedtuple(typename='CentralizedConfigs', field_names=['completed_epochs', 'completed_batches', 'num_training_examples'])
	SynchronousFedRoundConfigs = collections.namedtuple(typename='SynchronousFedRoundConfigs', field_names=['learners_per_round', 'completed_batches_per_round', 'completed_epochs_per_round'])

	MATPLOTLIB_MARKERS = [ ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "D", "d", "|", "_" ]
	MATPLOTLIB_MARKERS_ITER = itertools.cycle(MATPLOTLIB_MARKERS)

	MATPLOTLIB_LINESTYLES = [":", "-.", "--", "-"]
	MATPLOTLIB_LINESTYLES_ITER = itertools.cycle(MATPLOTLIB_LINESTYLES)

	MATPLOTLIB_HATCHES = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


	@classmethod
	def sort_alphanumeric(cls, iterable, reverse=True):
		""" Alphanumeric Sort: Sort the given iterable in the way that humans expect."""
		""" Taken from here: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python """
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(iterable, key=alphanum_key, reverse=reverse)  # reverse will return CPUs at the top

	@classmethod
	def atof(cls, text):
		"""https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside"""
		try:
			retval = float(text)
		except ValueError:
			retval = text
		return retval

	@classmethod
	def natural_keys(cls, text):
		"""
		https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
		alist.sort(key=natural_keys) sorts in human order
		http://nedbatchelder.com/blog/200712/human_sorting.html
		(See Toothy's implementation in the comments)
		float regex comes from https://stackoverflow.com/a/12643073/190597
		"""
		return [cls.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

	@classmethod
	def get_marker(cls, idx=None):
		if idx is None:
			return next(cls.MATPLOTLIB_MARKERS_ITER)
		else:
			return cls.MATPLOTLIB_MARKERS[idx]


	@classmethod
	def get_linestyle(cls, idx=None):
		if idx is None:
			return next(cls.MATPLOTLIB_LINESTYLES_ITER)
		else:
			return cls.MATPLOTLIB_LINESTYLES[idx]


	@classmethod
	def get_colors_map(cls, n, name='nipy_spectral'):
		"""A random color generator. Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
		RGB color; the keyword argument name must be a standard mpl colormap name."""
		"""
		Possible alternatives for name can be found here: https://matplotlib.org/tutorials/colors/colormaps.html
		"""
		return plt.cm.get_cmap(name, n)


	@classmethod
	def load_files_to_dict(cls, domain_files, centralized=False, federation_policies=True, filters=list()):

		assert(isinstance(domain_files, list))

		if centralized:
			centralized_json_objs = list()
			for filter in filters:
				centralized_json_objs.extend([json.load(open(file, 'r')) for file in domain_files if filter in file])

			return centralized_json_objs, [], []

		if federation_policies:
			centralized_policy_json_objs = [json.load(open(file, 'r')) for file in domain_files if 'centralized' in file]
			synchronous_policy_json_objs = [json.load(open(file, 'r')) for file in domain_files if 'synchronous_True' in file and 'centralized' not in file]
			# for file in domain_files:
			# 	if 'synchronous_False' in file and 'centralized' not in file:
			# 		print(file)
			# 		json.load(open(file, 'r'))
			asynchronous_policy_json_objs = [json.load(open(file, 'r')) for file in domain_files if 'synchronous_False' in file and 'centralized' not in file]
			return centralized_policy_json_objs, synchronous_policy_json_objs, asynchronous_policy_json_objs


	@classmethod
	def get_centralized_configs(cls, central_collection):

		fed_round= 'federation_round_0'
		hosts_results = central_collection[fed_round]['hosts_results']
		for host in hosts_results:
			completed_epochs = hosts_results[host]['completed_epochs']
			completed_batches = hosts_results[host]['completed_batches']
			num_training_examples = hosts_results[host]['num_training_examples']

		return cls.CentralizedConfigs(completed_epochs=completed_epochs, completed_batches=completed_batches, num_training_examples=num_training_examples)


	@classmethod
	def get_synchronous_fedround_configs(cls, sync_collection):
		synchronous_ts_scores = []
		fed_round = 'federation_round_0'
		hosts_results = sync_collection[fed_round]['hosts_results']
		learners_per_round = len(hosts_results)
		completed_epochs = 0
		completed_batches = 0

		for host in hosts_results:
			completed_epochs += hosts_results[host]['completed_epochs']
			completed_batches += hosts_results[host]['completed_batches']

		return cls.SynchronousFedRoundConfigs(learners_per_round=learners_per_round, completed_batches_per_round=completed_batches, completed_epochs_per_round=completed_epochs)


	@classmethod
	def process_ts_scores_by_time(cls, ts_scores_collection, max_execution_seconds=None):

		ts_scores_collection = sorted(ts_scores_collection, key=lambda x: x[0])
		first_ts_value = ts_scores_collection[0][0]
		ts_scores_collection = [(int((ts - first_ts_value) / 1000), score) for ts, score in ts_scores_collection]

		# Check if the collection is at least of size 2
		if len(ts_scores_collection) > 1:
			# Remove (0,0) from the plot. That is, if the policy has burn-in time, then set (x,0), where x is the first occurrence of community weights
			ts_scores_collection[0] = (ts_scores_collection[1][0]-1, 0)
		else:
			ts_scores_collection[0] = (0,0)

		new_ts_scores_collection = list()
		new_ts_scores_collection.append(ts_scores_collection[0])
		for idx, ts_score in enumerate(ts_scores_collection[1:]):
			prev_ts = new_ts_scores_collection[-1][0]
			current_ts = ts_score[0]

			if current_ts == prev_ts:
				del new_ts_scores_collection[-1]
				new_ts_scores_collection.append(ts_score)
			else:
				new_ts_scores_collection.append(ts_score)
		# print(ts_scores_collection)

		if max_execution_seconds is not None:

			if len(new_ts_scores_collection) > 1:
				new_ts_scores_collection = [ts_score for ts_score in new_ts_scores_collection if ts_score[0] <= max_execution_seconds]

				last_value = new_ts_scores_collection[-1][0]
				burning_time = int(max_execution_seconds - last_value)
				# print(burning_time)
				new_ts_scores_collection = [(burning_time, 0)] + [(ts_score[0] + burning_time, ts_score[1]) for ts_score in new_ts_scores_collection]
				# print(new_ts_scores_collection)
				# print()

		return new_ts_scores_collection


	@classmethod
	def get_centralized_test_evaluations(cls, central_collection, metric, test_dataset_eval=True, max_execution_seconds=None):
		centralized_ts_scores = []
		for fed_round in central_collection:
			hosts_results = central_collection[fed_round]['hosts_results']
			for host in hosts_results:
				if test_dataset_eval is True:
					evaluations = hosts_results[host]['test_set_evaluations']
				else:
					evaluations = hosts_results[host]['train_set_evaluations']
				for evaluation in evaluations:
					centralized_ts_scores.append((evaluation['unix_ms'], evaluation['evaluation_results'][metric]))
		centralized_ts_scores = cls.process_ts_scores_by_time(centralized_ts_scores, max_execution_seconds=max_execution_seconds)

		return centralized_ts_scores


	@classmethod
	def get_synchronous_community_test_evaluations(cls, sync_collection, metric, max_execution_seconds=None):

		synchronous_ts_scores = []
		for fed_round in sync_collection:
			fed_round_accuracy = sync_collection[fed_round]['fedround_evaluation']['evaluation_results'][metric]
			fed_round_score_timestamp = sync_collection[fed_round]['fedround_evaluation']['unix_ms']
			synchronous_ts_scores.append((fed_round_score_timestamp, fed_round_accuracy))
		synchronous_ts_scores = cls.process_ts_scores_by_time(synchronous_ts_scores, max_execution_seconds=max_execution_seconds)

		return synchronous_ts_scores


	@classmethod
	def get_asynchronous_community_test_evaluations(cls, async_collection, metric, max_execution_seconds=None):
		asynchronous_ts_scores = []
		training_start_time = -float("inf")
		for fed_round in async_collection:
			hosts_results = async_collection[fed_round]['hosts_results']
			for host in hosts_results:
				training_start_time = min(training_start_time, int(hosts_results[host]['compute_init_unix_time']))
				test_set_evaluations = hosts_results[host]['test_set_evaluations']
				for evaluation in test_set_evaluations:
					if evaluation['is_after_community_update'] is True:
						timestamp = int(evaluation['unix_ms'])
						score_value = float(evaluation['evaluation_results'][metric])
						asynchronous_ts_scores.append((timestamp, score_value))
		# fed_round_score = async_collection[fed_round]['fedround_evaluation']['evaluation_results'][metric]
		# fed_round_score_timestamp = async_collection[fed_round]['fedround_evaluation']['unix_ms']
		# asynchronous_ts_scores.append((fed_round_score_timestamp, fed_round_score))
		asynchronous_ts_scores.append((training_start_time, 0))
		asynchronous_ts_scores = cls.process_ts_scores_by_time(asynchronous_ts_scores, max_execution_seconds)

		return asynchronous_ts_scores


	@classmethod
	def get_synchronous_community_test_evaluations_local_epochs(cls, sync_collection, metric, max_execution_seconds=None):
		synchronous_local_epochs_scores = []
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(sync_collection, reverse=False)
		total_execution_time = 0
		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		for fed_round in sorted_fedrounds_ids:
			fed_round_accuracy = sync_collection[fed_round]['fedround_evaluation']['evaluation_results'][metric]
			hosts_results = sync_collection[fed_round]['hosts_results']
			total_local_epochs = 0
			for host in hosts_results:
				total_local_epochs += hosts_results[host]['completed_epochs']
			for host in hosts_results:
				host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
				host_processing_time = np.sum([host_epochs_times])
				host_processing_time /= 1000
				total_execution_time += host_processing_time
			if total_execution_time <= max_execution_seconds:
				if len(synchronous_local_epochs_scores) == 0:
					synchronous_local_epochs_scores.append((total_local_epochs,fed_round_accuracy))
				else:
					previous_local_epochs = synchronous_local_epochs_scores[-1][0]
					synchronous_local_epochs_scores.append((previous_local_epochs+total_local_epochs, fed_round_accuracy))
		return synchronous_local_epochs_scores


	@classmethod
	def get_synchronous_community_test_evaluations_global_epochs(cls, sync_collection, metric, max_execution_seconds=None, delay=60):
		synchronous_global_epochs_scores = []
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(sync_collection, reverse=False)
		init_timestamp = sync_collection["federation_round_0"]['fedround_evaluation']['unix_ms']
		total_execution_time = 0
		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		for fed_round in sorted_fedrounds_ids:
			fed_round_accuracy = sync_collection[fed_round]['fedround_evaluation']['evaluation_results'][metric]
			fed_round_timestamp = sync_collection[fed_round]['fedround_evaluation']['unix_ms']
			hosts_results = sync_collection[fed_round]['hosts_results']
			total_global_epochs = float("inf")
			for host in hosts_results:
				total_global_epochs = min(total_global_epochs, hosts_results[host]['completed_epochs'])
			# for host in hosts_results:
			# 	host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
			# 	host_processing_time = np.sum([host_epochs_times])
			# 	host_processing_time /= 1000
			# 	total_execution_time += host_processing_time
			fedround_wall_clock_time = np.divide(sync_collection[fed_round]["fedround_execution_time_ms"], 1000)
			fedround_wall_clock_time += delay
			total_execution_time += fedround_wall_clock_time
			if total_execution_time <= max_execution_seconds:
				if len(synchronous_global_epochs_scores) == 0:
					synchronous_global_epochs_scores.append((total_global_epochs,fed_round_accuracy))
				else:
					previous_global_epochs = synchronous_global_epochs_scores[-1][0]
					synchronous_global_epochs_scores.append((previous_global_epochs+total_global_epochs, fed_round_accuracy))
		return synchronous_global_epochs_scores


	@classmethod
	def get_asynchronous_community_test_evaluations_local_epochs(cls, async_collection, metric, max_execution_seconds=None):
		asynchronous_ts_scores_with_update = []
		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		for fed_round in async_collection:
			hosts_results = async_collection[fed_round]['hosts_results']
			for host in hosts_results:
				training_start_time = int(hosts_results[host]['compute_init_unix_time'])
				test_set_evaluations = hosts_results[host]['test_set_evaluations']
				for evaluation in test_set_evaluations:
					current_timestamp = int(evaluation['unix_ms'])
					score_value = float(evaluation['evaluation_results'][metric])
					current_duration_sec = (current_timestamp - training_start_time) / 1000
					if current_duration_sec < max_execution_seconds:
						if evaluation['is_after_community_update'] is True:
							asynchronous_ts_scores_with_update.append((current_timestamp, score_value, True))
						else:
							asynchronous_ts_scores_with_update.append((current_timestamp, score_value, False))

		asynchronous_ts_scores_with_update = sorted(asynchronous_ts_scores_with_update, key=lambda x: x[0])

		asynchronous_local_epochs_scores = []
		local_epochs = 0
		previous_with_update = False
		max_score = -float("inf")
		for record in asynchronous_ts_scores_with_update:
			score = record[1]
			if record[2] is False:
				local_epochs += 1
				previous_with_update = False
				max_score = score if score > max_score else max_score
			else:
				if previous_with_update:
					local_epochs += 1
				asynchronous_local_epochs_scores.append((local_epochs, score))
				max_score = -float("inf")
				previous_with_update = True

		return asynchronous_local_epochs_scores


	@classmethod
	def get_asynchronous_community_test_evaluations_global_epochs(cls, async_collection, metric, max_execution_seconds=None):
		asynchronous_ts_scores_with_update = []
		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		for fed_round in async_collection:
			hosts_results = async_collection[fed_round]['hosts_results']
			for host in hosts_results:
				training_start_time = int(hosts_results[host]['compute_init_unix_time'])
				test_set_evaluations = hosts_results[host]['test_set_evaluations']
				for evaluation in test_set_evaluations:
					current_timestamp = int(evaluation['unix_ms'])
					score_value = float(evaluation['evaluation_results'][metric])
					current_duration_sec = (current_timestamp - training_start_time) / 1000
					if current_duration_sec < max_execution_seconds:
						if evaluation['is_after_community_update'] is False:
							asynchronous_ts_scores_with_update.append((host, current_timestamp, score_value, False))
						else:
							asynchronous_ts_scores_with_update.append((host, current_timestamp, score_value, True))

		asynchronous_ts_scores_with_update = sorted(asynchronous_ts_scores_with_update, key=lambda x: x[1])
		original_hosts_set = set([x[0] for x in asynchronous_ts_scores_with_update])
		global_epoch_reached_host_set = copy.deepcopy(original_hosts_set)
		asynchronous_global_epochs_scores = []

		max_so_far = -float("inf")
		# scores = []
		scores = collections.defaultdict(int)
		for record in asynchronous_ts_scores_with_update:
			host = record[0]
			score = record[2]
			is_community = record[3]
			if is_community:
				max_so_far = max(score, max_so_far)
			else:
				scores[host] += 1
			if host in global_epoch_reached_host_set:
				global_epoch_reached_host_set.remove(host)
			if len(global_epoch_reached_host_set) == 0 and is_community:
				# What is the minimum number of local epochs completed?
				# This is our number for global epochs
				global_epochs = min(scores.values())
				asynchronous_global_epochs_scores.append((global_epochs, max_so_far))
				global_epoch_reached_host_set = copy.deepcopy(original_hosts_set)
				max_so_far = -float("inf")
				# scores = []
				# scores = collections.defaultdict(int)

		return asynchronous_global_epochs_scores


	@classmethod
	def get_learner_plot_device_key(cls, device_id, host_name=None):
		device_id = ':'.join(device_id.split('/')[-1].split(':')[-2:]).upper().replace(']','')
		device_id = device_id.split(":")[0]
		if host_name is None:
			return device_id
		else:
			ports = re.findall(r'isi\.edu:\d+',host_name)
			ps_port = ports[0].split(":")[1]
			ws_port = ports[1].split(":")[1]
			if 'bdnf' in host_name:
				return '{}:bdnf:{}'.format(device_id, ws_port)
			else:
				return '{}:learn:{}'.format(device_id, ws_port)


	@classmethod
	def get_learners_keys(cls, json_federated_rounds_obj, with_host_name=False, sort=True):
		first_round_id = list(json_federated_rounds_obj.keys())[0]
		fdr_hosts = json_federated_rounds_obj[first_round_id]['hosts_results']
		learners_keys = list()
		for host_id in fdr_hosts:
			host_data = fdr_hosts[host_id]
			if with_host_name:
				device_id = cls.get_learner_plot_device_key(host_data['training_devices'], host_name=host_data['host_id'])  # GPU or CPU
			else:
				device_id = cls.get_learner_plot_device_key(host_data['training_devices'], host_name=None)  # GPU or CPU
			learners_keys.append(device_id)
		if sort:
			return LogsProcessingUtil.sort_alphanumeric(learners_keys)
		else:
			return learners_keys


	@classmethod
	def get_experiment_maximum_execution_seconds(cls, filename):
		if 'targetexectimemins' in filename:
			max_exec_secs = [token for token in filename.split('.') if 'targetexectimemins' in token][0].split('_')[1]
			max_exec_secs = int(max_exec_secs) * 60
			return max_exec_secs
		else:
			return None


	@classmethod
	def heatmap(cls, data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
		"""
		Create a heatmap from a numpy array and two lists of labels.

		Arguments:
			data       : A 2D numpy array of shape (N,M)
			row_labels : A list or array of length N with the labels
						 for the rows
			col_labels : A list or array of length M with the labels
						 for the columns
		Optional arguments:
			ax         : A matplotlib.axes.Axes instance to which the heatmap
						 is plotted. If not provided, use current axes or
						 create a new one.
			cbar_kw    : A dictionary with arguments to
						 :meth:`matplotlib.Figure.colorbar`.
			cbarlabel  : The label for the colorbar
		All other arguments are directly passed on to the imshow call.
		"""

		if not ax:
			ax = plt.gca()

		# Plot the heatmap
		im = ax.imshow(data, **kwargs)

		# Create colorbar
		cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
		cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

		# We want to show all ticks...
		ax.set_xticks(np.arange(data.shape[1]))
		ax.set_yticks(np.arange(data.shape[0]))
		# ... and label them with the respective list entries.
		ax.set_xticklabels(col_labels)
		ax.set_yticklabels(row_labels)

		# Let the horizontal axes labeling appear on top.
		ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

		# Rotate the tick labels and set their alignment.
		plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

		# Turn spines off and create white grid.
		for edge, spine in ax.spines.items():
			spine.set_visible(False)

		ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
		ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
		ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
		ax.tick_params(which="minor", bottom=False, left=False)

		return im, cbar


	@classmethod
	def annotate_heatmap(cls, im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
		"""
		A function to annotate a heatmap.

		Arguments:
			im         : The AxesImage to be labeled.
		Optional arguments:
			data       : Data used to annotate. If None, the image's data is used.
			valfmt     : The format of the annotations inside the heatmap.
						 This should either use the string format method, e.g.
						 "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
			textcolors : A list or array of two color specifications. The first is
						 used for values below a threshold, the second for those
						 above.
			threshold  : Value in data units according to which the colors from
						 textcolors are applied. If None (the default) uses the
						 middle of the colormap as separation.

		Further arguments are passed on to the created text labels.
		"""

		if not isinstance(data, (list, np.ndarray)):
			data = im.get_array()

		# Normalize the threshold to the images color range.
		if threshold is not None:
			threshold = im.norm(threshold)
		else:
			threshold = im.norm(data.max()) / 2.

		# Set default alignment to center, but allow it to be
		# overwritten by textkw.
		kw = dict(horizontalalignment="center",
				  verticalalignment="center")
		kw.update(textkw)

		# Get the formatter in case a string is supplied
		if isinstance(valfmt, str):
			valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

		# Loop over the data and create a `Text` for each "pixel".
		# Change the text's color depending on the data.
		texts = []
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
				text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
				texts.append(text)

		return texts


	@classmethod
	def get_labels_from_filename(cls, filename, centralized=False, synchronous=False):
		file_tokens = filename.split('.')
		line_labels = []

		clients_token = [token for token in file_tokens if 'client' in token]
		if len(clients_token) > 0:
			clients_token = clients_token[0].replace('clients_', '')
		else:
			clients_token = None

		community_function_token = [token for token in file_tokens if 'Function' in token]
		if len(community_function_token) > 0:
			community_function_token = community_function_token[0].replace('Function_', '')
		else:
			community_function_token = None
		line_labels.append(community_function_token)

		lr_token = [token for token in file_tokens if 'learningrate' in token or 'lr_' in token][0].replace('lr_', 'lr:').replace('learningrate_', 'lr:').replace('lr:0', 'lr:0.')
		line_labels.append(lr_token)
		momentum_token = [token for token in file_tokens if 'SGDWithMomentum' in token or 'AdamOpt' in token][0].replace('SGDWithMomentum', 'm:').replace('AdamOpt', 'Adam')
		momentum_tokens = momentum_token.split(":")
		if momentum_tokens[1][0] == "0" and momentum_tokens[1][-1] != "0":
			line_labels.append(''.join([momentum_tokens[0], ":", momentum_tokens[1][0], ".", momentum_tokens[1][1:]]))
		batchsize_token = [token for token in file_tokens if 'batchsize' in token or 'b_' in token][0].replace('batchsize_', 'b:').replace('b_', 'b:')
		line_labels.append(batchsize_token)

		holdout_token = ''
		burnin_token = ''
		ufrequency_token = ''
		min_vloss = ''
		max_vloss = ''
		delta_le = ''
		is_adaptive = any([True for x in file_tokens if 'VLossChange' in x])
		if centralized is False:
			if synchronous:
				ufrequency_token = [token for token in file_tokens if 'UFREQUENCY' in token][0].replace('UFREQUENCY=', 'uf:')
				line_labels.append(ufrequency_token)
			else:
				if is_adaptive:
					line_labels.append(ufrequency_token)
					holdout_token = [token for token in file_tokens if 'StratifiedHoldout' in token][0].replace('StratifiedHoldout', 'Holdout:').replace('pct', '%')
					line_labels.append(holdout_token)
					ufrequency_token = [token for token in file_tokens if 'UFREQUENCY' in token][0].replace('UFREQUENCY=', 'uf:')
					min_max_vloss = ufrequency_token.split('VLossChange')[1]
					min_max_vloss = min_max_vloss.split("to")
					min_vloss = min_max_vloss[0]
					max_vloss = min_max_vloss[1]
					delta_le = [token for token in file_tokens if 'deltaLE' in token][0]
					delta_le = delta_le.split('_')[1]
				else:
					ufrequency_token = [token for token in file_tokens if 'UFREQUENCY' in token][0].replace('UFREQUENCY=', 'uf:')
					line_labels.append(ufrequency_token)
					# burnin_token = [token for token in file_tokens if 'BURNING' in token][0].replace('BURNING_EPOCHS=', 'burnin:')
					# line_labels.append(burnin_token)


		filename_labels = LogsProcessingUtil.FilenameLabels(clients=clients_token, community_function=community_function_token, learning_rate=lr_token,
															sgd_momentum=momentum_token, batchsize=batchsize_token, ufrequency=ufrequency_token,
															holdout=holdout_token, burnin=burnin_token, min_vloss=min_vloss, max_vloss=max_vloss, delta_le=delta_le,
															is_adaptive=is_adaptive)

		return line_labels, filename_labels


	@classmethod
	def compute_global_training_steps(cls, policy_collection, synchronous_policy=True,
									  exclude_unix_ms=True, exclude_score=True,
									  max_execution_seconds=15000, delay=60):

		total_batches = 0
		total_batches_with_scores = list()
		if synchronous_policy:
			sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
			total_execution_time = 0
			for fed_round in sorted_fedrounds_ids:
				hosts_results = policy_collection[fed_round]['hosts_results']
				max_processing_time = -float("inf")
				for host in hosts_results:
					host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
					host_processing_time = np.sum([host_epochs_times])
					host_processing_time /= 1000
					max_processing_time = max(max_processing_time, host_processing_time)
				# total_execution_time += max_processing_time

				hosts_results = policy_collection[fed_round]['hosts_results']
				for host in hosts_results:
					total_batches += hosts_results[host]['completed_batches']
				fed_round_accuracy = policy_collection[fed_round]['fedround_evaluation']['evaluation_results']['accuracy']
				fedround_wall_clock_time = np.divide(policy_collection[fed_round]["fedround_execution_time_ms"], 1000)
				fedround_wall_clock_time += delay
				total_execution_time += fedround_wall_clock_time
				if total_execution_time <= max_execution_seconds:
					total_batches_with_scores.append((total_batches, fed_round_accuracy))

		else:
			init_time_ms = float("inf")
			first_round_hosts_data = policy_collection['federation_round_0']['hosts_results']
			for host in first_round_hosts_data:
				init_time_ms = min(init_time_ms, first_round_hosts_data[host]['compute_init_unix_time'])

			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				for host in hosts_results:
					batch_size = int( (hosts_results[host]['completed_epochs'] * hosts_results[host]['num_training_examples']) / hosts_results[host]['completed_batches'])
					batches_per_epoch = int(hosts_results[host]['num_training_examples'] / batch_size)
					test_set_evaluations = hosts_results[host]['test_set_evaluations']
					total_batches += hosts_results[host]['completed_batches']
					host_batches = 0
					for evaluation in test_set_evaluations:
						if evaluation['is_after_community_update'] is False:
							host_batches += batches_per_epoch
						else:
							timestamp = float(evaluation['unix_ms'])
							score = float(evaluation['evaluation_results']['accuracy'])
							total_batches_with_scores.append((timestamp, host_batches, score))
							host_batches = 0
			total_batches_with_scores = sorted(total_batches_with_scores, key=lambda x: x[0])
			total_batches_with_scores = [( (x[0] - total_batches_with_scores[0][0])/1000, x[1], x[2])
													   for x in total_batches_with_scores]
			total_batches_with_scores = [x for x in total_batches_with_scores
													   if x[0] <= max_execution_seconds]

			if exclude_unix_ms and exclude_score:
				total_batches_with_scores = [x[1] for x in total_batches_with_scores]
			if exclude_unix_ms and not exclude_score:
				total_batches_with_scores = [(x[1],x[2]) for x in total_batches_with_scores]
			if not exclude_unix_ms and exclude_score:
				total_batches_with_scores = [(x[0],x[1]) for x in total_batches_with_scores]

		return total_batches, total_batches_with_scores


	@classmethod
	def get_local_epochs_per_learner(cls, filepath, exclude_unix_ms=True, exclude_score=True):

		_, _, asynchronous = LogsProcessingUtil.load_files_to_dict([filepath], federation_policies=True)
		asynchronous = asynchronous[0]
		if 'federation_rounds_results' in asynchronous:
			asynchronous = asynchronous['federation_rounds_results']

		hosts_results = asynchronous['federation_round_0']['hosts_results']
		learners_le = dict()

		for host_id in sorted(hosts_results):
			total_updates = 0
			total_epochs = 0
			epochs_within_updates_counter = 0
			epochs_within_updates_counter_list = list()

			for test_set_evaluation in hosts_results[host_id]['test_set_evaluations']:
				if test_set_evaluation['is_after_community_update'] is True:
					timestamp = test_set_evaluation['unix_ms']
					accuracy = test_set_evaluation['evaluation_results']['accuracy']
					epochs_within_updates_counter_list.append((timestamp, epochs_within_updates_counter, accuracy))
					epochs_within_updates_counter = 0
					total_updates += 1
				if test_set_evaluation['is_after_epoch_completion'] is True:
					total_epochs += 1
					epochs_within_updates_counter += 1

			epochs_within_updates_counter_list = sorted(epochs_within_updates_counter_list, key=lambda x: x[0])
			if exclude_unix_ms and exclude_score:
				epochs_within_updates_counter_list = [x[1] for x in epochs_within_updates_counter_list]
			if exclude_unix_ms and not exclude_score:
				epochs_within_updates_counter_list = [(x[1],x[2]) for x in epochs_within_updates_counter_list]
			if not exclude_unix_ms and exclude_score:
				epochs_within_updates_counter_list = [(x[0],x[1]) for x in epochs_within_updates_counter_list]
			learners_le[host_id] = epochs_within_updates_counter_list

		return learners_le


	@classmethod
	def analyze_learners_model_staleness_based_on_community_requests(cls, file, exclude_unix_ms=True, exclude_score=True):

		_, _, async_collections = LogsProcessingUtil.load_files_to_dict([file], federation_policies=True)
		async_collection = async_collections[0]
		async_collection = async_collection['federation_rounds_results'] \
			if "federation_rounds_results" in async_collection else async_collection
		hosts_results = async_collection['federation_round_0']['hosts_results']
		learners_ids = set(hosts_results)

		learners_evaluation_collection = []
		for learner_id in hosts_results:
			learner_test_evaluations = hosts_results[learner_id]['test_set_evaluations']
			for test_evaluation in learner_test_evaluations:
				if test_evaluation['is_after_community_update'] is True:
					evaluation_timestamp = test_evaluation['unix_ms']
					accuracy = test_evaluation['evaluation_results']['accuracy']
					learners_evaluation_collection.append([evaluation_timestamp, learner_id, accuracy])

		learners_evaluation_collection = sorted(learners_evaluation_collection, key=lambda x: x[0])

		# Init global and learners update scalar clock
		global_clock = 0
		learners_last_global_clock = dict()
		for learner_id in learners_ids:
			learners_last_global_clock[learner_id] = 0

		learners_staleness = defaultdict(list)
		for learner_eval in learners_evaluation_collection:
			learner_eval_timestamp = learner_eval[0]
			learner_eval_id = learner_eval[1]
			learner_eval_accuracy = learner_eval[2]
			staleness = global_clock - learners_last_global_clock[learner_eval_id]
			if exclude_unix_ms and exclude_score:
				learners_staleness[learner_eval_id].append(staleness)
			if exclude_unix_ms and not exclude_score:
				learners_staleness[learner_eval_id].append((staleness, learner_eval_accuracy))
			if not exclude_unix_ms and exclude_score:
				learners_staleness[learner_eval_id].append((learner_eval_timestamp, staleness))
			if not exclude_unix_ms and not exclude_score:
				learners_staleness[learner_eval_id].append((learner_eval_timestamp, staleness, learner_eval_accuracy))
			global_clock += 1
			learners_last_global_clock[learner_eval_id] = global_clock

		return learners_staleness

	@classmethod
	def analyze_learners_model_staleness_based_on_community_steps(cls, file, exclude_unix_ms=True, exclude_score=True):

		_, _, async_collections = LogsProcessingUtil.load_files_to_dict([file], federation_policies=True)
		async_collection = async_collections[0]
		if 'federation_rounds_results' in async_collection:
			hosts_results = async_collection['federation_rounds_results']['federation_round_0']['hosts_results']
		else:
			hosts_results = async_collection['federation_round_0']['hosts_results']
		learners_ids = set(hosts_results)

		learners_evaluation_collection = []
		for learner_id in hosts_results:
			completed_batches = int(hosts_results[learner_id]["completed_batches"])
			completed_epochs = int(hosts_results[learner_id]["completed_epochs"])
			steps_per_epoch = int(np.ceil(completed_batches / completed_epochs))
			learner_test_evaluations = hosts_results[learner_id]['test_set_evaluations']
			steps_in_between_community_requests = 0
			for test_evaluation in learner_test_evaluations:
				if test_evaluation['is_after_community_update'] is True:
					evaluation_timestamp = test_evaluation['unix_ms']
					accuracy = test_evaluation['evaluation_results']['accuracy']
					learners_evaluation_collection.append([evaluation_timestamp, learner_id, steps_in_between_community_requests, accuracy])
					steps_in_between_community_requests = 0
				else:
					steps_in_between_community_requests += steps_per_epoch

		learners_evaluation_collection = sorted(learners_evaluation_collection, key=lambda x: x[0])

		# Init global and learners update scalar clock
		global_update_steps = 0
		learners_last_global_update_steps = dict()
		for learner_id in learners_ids:
			learners_last_global_update_steps[learner_id] = 0

		learners_staleness = defaultdict(list)
		for learner_eval in learners_evaluation_collection:
			learner_eval_timestamp = learner_eval[0]
			learner_eval_id = learner_eval[1]
			learner_completed_steps = learner_eval[2]
			learner_eval_accuracy = learner_eval[3]
			staleness = global_update_steps - learners_last_global_update_steps[learner_eval_id] + learner_completed_steps
			if exclude_unix_ms and exclude_score:
				learners_staleness[learner_eval_id].append(staleness)
			if exclude_unix_ms and not exclude_score:
				learners_staleness[learner_eval_id].append((staleness, learner_eval_accuracy))
			if not exclude_unix_ms and exclude_score:
				learners_staleness[learner_eval_id].append((learner_eval_timestamp, staleness))
			if not exclude_unix_ms and not exclude_score:
				learners_staleness[learner_eval_id].append((learner_eval_timestamp, staleness, learner_eval_accuracy))
			global_update_steps += learner_completed_steps
			learners_last_global_update_steps[learner_eval_id] = global_update_steps

		return learners_staleness


	@classmethod
	def compute_local_global_training_epochs(cls, policy_collection, synchronous_policy=True, max_execution_seconds=None):

		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		local_epochs = 0
		global_epochs = 0
		if synchronous_policy:
			total_execution_time = 0
			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				for host in hosts_results:
					host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
					host_processing_time = np.sum([host_epochs_times])
					host_processing_time /= 1000
					total_execution_time += host_processing_time

				if total_execution_time <= max_execution_seconds:
					hosts_results = policy_collection[fed_round]['hosts_results']
					fed_round_global_epochs = float("inf")
					for host in hosts_results:
						local_epochs += hosts_results[host]['completed_epochs']
						fed_round_global_epochs = min(fed_round_global_epochs, hosts_results[host]['completed_epochs'])
					global_epochs += fed_round_global_epochs
		else:
			init_time_ms = float("inf")
			first_round_hosts_data = policy_collection['federation_round_0']['hosts_results']
			for host in first_round_hosts_data:
				init_time_ms = min(init_time_ms, first_round_hosts_data[host]['compute_init_unix_time'])

			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				hosts_epochs = []
				for host in hosts_results:
					current_host_epochs = 0
					train_set_evaluations = hosts_results[host]['train_set_evaluations']
					for evaluation in train_set_evaluations:
						if evaluation['is_after_epoch_completion'] is True:
							end_time_ms = evaluation['unix_ms']
							if TimeUtil.delta_diff_in_secs(end_time_ms, init_time_ms) < max_execution_seconds:
								local_epochs += 1
								current_host_epochs += 1
					hosts_epochs.append(current_host_epochs)
				global_epochs = min(hosts_epochs)

		return local_epochs, global_epochs


	@classmethod
	def compute_update_requests(cls, policy_collection, synchronous_policy=True, max_execution_seconds=None,
								exclude_unix_ms=True, exclude_score=False, delay=60):

		if max_execution_seconds is None:
			max_execution_seconds = float("inf")

		total_update_requests = 0
		update_requests_with_scores = list()
		if synchronous_policy:
			total_execution_time = 0
			sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
			for fed_round in sorted_fedrounds_ids:
				hosts_results = policy_collection[fed_round]['hosts_results']
				# max_processing_time = -float("inf")
				# for host in hosts_results:
				# 	host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
				# 	host_processing_time = np.sum([host_epochs_times])
				# 	host_processing_time /= 1000
				# 	max_processing_time = max(max_processing_time, host_processing_time)
				# total_execution_time += max_processing_time

				fedround_wall_clock_time = np.divide(policy_collection[fed_round]["fedround_execution_time_ms"], 1000)
				fedround_wall_clock_time += delay
				total_execution_time += fedround_wall_clock_time

				if total_execution_time <= max_execution_seconds:
					hosts_results = policy_collection[fed_round]['hosts_results']
					number_of_hosts = len(hosts_results)
					total_update_requests += number_of_hosts
					fed_round_accuracy = policy_collection[fed_round]['fedround_evaluation']['evaluation_results']['accuracy']
					update_requests_with_scores.append((total_update_requests, fed_round_accuracy))
		else:
			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				for host in hosts_results:
					test_set_evaluations = hosts_results[host]['test_set_evaluations']
					for evaluation in test_set_evaluations:
						if evaluation['is_after_community_update'] is True:
							score_value = float(evaluation['evaluation_results']['accuracy'])
							score_value_ms = float(evaluation['unix_ms'])
							total_update_requests += 1
							update_requests_with_scores.append((score_value_ms, score_value))

			update_requests_with_scores = sorted(update_requests_with_scores, key=lambda x: x[0])
			update_requests_with_scores = [((x[0]-update_requests_with_scores[0][0])/1000, x[1]) for x in update_requests_with_scores]
			update_requests_with_scores = [(x[0], x[1]) for x in update_requests_with_scores if x[0] <= max_execution_seconds]
			update_requests_with_scores = [ (idx,score[1]) for idx, score in enumerate(update_requests_with_scores)]

		return total_update_requests, update_requests_with_scores


	@classmethod
	def compute_epochs_processing_time_aux(cls, policy_collection, synchronous_policy=True, max_execution_seconds=None):
		total_execution_time = 0
		updates_processing_time = list()
		if synchronous_policy:
			sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
			for fed_round in sorted_fedrounds_ids:
				hosts_results = policy_collection[fed_round]['hosts_results']
				max_processing_time = -float("inf")
				for host in hosts_results:
					host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
					host_processing_time = np.sum([host_epochs_times])
					host_processing_time /= 1000
					max_processing_time = max(max_processing_time, host_processing_time)

				# max_processing_time = sum(22*[len(host_epochs_times)])
				# print(len(host_epochs_times))
				# print(max_processing_time)
				total_execution_time += max_processing_time
				if total_execution_time <= max_execution_seconds:
					max_processing_time_point = updates_processing_time[-1][0] + max_processing_time if len(updates_processing_time) > 0 else max_processing_time
					fed_round_accuracy = policy_collection[fed_round]['fedround_evaluation']['evaluation_results']['accuracy']
					updates_processing_time.append((max_processing_time_point, fed_round_accuracy))
		else:
			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				max_processing_time = -float("inf")
				max_processing_epochs = []
				gpu_times = []
				cpu_times = []
				for host in hosts_results:
					host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
					all_epoch_times = [x / 1000 for x in host_epochs_times]
					if 'bdnf' in host:
						gpu_times.extend(all_epoch_times)
					else:
						cpu_times.extend(all_epoch_times)
					host_processing_time = np.sum(host_epochs_times)
					host_processing_time /= 1000
					# Assign the processing time that was the maximum across all learners
					if host_processing_time > max_processing_time:
						max_processing_time = host_processing_time
						max_processing_epochs = host_epochs_times

					# Create Processing time collection based on community updates
					test_set_evaluations = hosts_results[host]['test_set_evaluations']
					previous_epoch_idx = 0
					current_processing_time = 0
					for idx, evaluation in enumerate(test_set_evaluations):
						score_value = float(evaluation['evaluation_results']['accuracy'])
						if evaluation['is_after_community_update'] is True:
							current_processing_time = current_processing_time + int(np.ceil(sum(host_epochs_times[previous_epoch_idx:idx-1])/1000))
							previous_epoch_idx = idx
							updates_processing_time.append([current_processing_time, score_value])

				gpu_average = np.average(gpu_times)
				cpu_average = np.average(cpu_times)

				# Find total execution time
				max_processing_epochs = [etime/1000 for etime in max_processing_epochs]
				for etime in max_processing_epochs:
					total_execution_time += etime
					if total_execution_time > max_execution_seconds:
						total_execution_time = total_execution_time - etime

				# Find processing epochs and community update value
				updates_processing_time = sorted(updates_processing_time, key=lambda x: (x[0],x[1]))
				ordered_dict = collections.OrderedDict()
				for t,s in updates_processing_time:
					if t < max_execution_seconds:
						if t in ordered_dict:
							ordered_dict[t] = max(ordered_dict[t], s)
						else:
							ordered_dict[t] = s
				updates_processing_time = ordered_dict.items()


		return total_execution_time, updates_processing_time


	@classmethod
	def compute_update_frequency(cls, policy_collection):
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
		for fed_round in sorted_fedrounds_ids:
			hosts_results = policy_collection[fed_round]['hosts_results']
			all_hosts_local_epochs = []
			for host in hosts_results:
				train_set_evaluations = hosts_results[host]['train_set_evaluations']
				host_local_epochs = []
				host_cycle_epochs = 0
				for evaluation in train_set_evaluations:
					if evaluation['is_after_epoch_completion'] is True:
						host_cycle_epochs += 1
					else:
						host_local_epochs.append(host_cycle_epochs)
						host_cycle_epochs = 0
				all_hosts_local_epochs.append(int(stats.mode(host_local_epochs)[0][0]))


	@classmethod
	def get_learners_federation_losses_by_processing_time(cls, policy_collection):
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
		ts_proc_is_local_epoch = []
		for fed_round in sorted_fedrounds_ids:
			hosts_results = policy_collection[fed_round]['hosts_results']
			for host in hosts_results:
				host_ts_proc_is_local_epoch = []
				host_epochs_times_ms = hosts_results[host]['epochs_exec_times_ms']
				host_epochs_times_sec = [x / 1000 for x in host_epochs_times_ms]
				print(host, np.mean(host_epochs_times_sec))
				test_set_evaluations = hosts_results[host]['test_set_evaluations']
				# Left and Right indexes associate with the slice of epoch times that we need
				left_idx = 0
				right_idx = 0
				previous_timestamp = -1
				current_timestamp = -1
				for eval_idx, evaluation in enumerate(test_set_evaluations):
					if evaluation['is_after_community_update'] is False:
						previous_timestamp = current_timestamp
						current_timestamp = evaluation['unix_ms']
						right_idx += 1
					if evaluation['is_after_community_update'] is True:
						processing_time = sum(host_epochs_times_sec[left_idx:right_idx])
						host_ts_proc_is_local_epoch.append((previous_timestamp, processing_time, True))
						left_idx = right_idx
				ts_proc_is_local_epoch.extend(host_ts_proc_is_local_epoch)

			metis_grpc_evaluator_metadata = policy_collection[fed_round]['metis_grpc_evaluator_metadata']
			if 'evaluation_requests' in metis_grpc_evaluator_metadata:
				federation_loss_requests = metis_grpc_evaluator_metadata['evaluation_requests']
			else:
				federation_loss_requests = metis_grpc_evaluator_metadata['metis_grpc_evaluator_metadata']['evaluation_requests']
			for request in federation_loss_requests:
				if ("is_community_model" in request and request["is_community_model"] is False) or \
					"is_community_model" not in request:
					request_ts = request['request_unix_time']
					ts_proc_is_local_epoch.append((request_ts, 0, False))

		ts_proc_is_local_epoch = sorted(ts_proc_is_local_epoch)
		sum_proc_time = 0
		fvl_proc_time = []
		for tpl in ts_proc_is_local_epoch:
			proc_time = tpl[1]
			is_local_epoch = tpl[2]
			if is_local_epoch:
				sum_proc_time += proc_time
			else:
				fvl_proc_time.append("{:.2f}".format(sum_proc_time))
		return fvl_proc_time


	@classmethod
	def get_learners_federation_losses_by_wall_clock_time(cls, policy_collection):
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
		fvl_wall_clock_time = []
		for fed_round in sorted_fedrounds_ids:
			metis_grpc_evaluator_metadata = policy_collection[fed_round]['metis_grpc_evaluator_metadata']
			if 'evaluation_requests' in metis_grpc_evaluator_metadata:
				federation_loss_requests = metis_grpc_evaluator_metadata['evaluation_requests']
			else:
				federation_loss_requests = metis_grpc_evaluator_metadata['metis_grpc_evaluator_metadata']['evaluation_requests']
			for request in federation_loss_requests:
				if ("is_community_model" in request and request["is_community_model"] is False) or \
					"is_community_model" not in request:
					request_ts = request['request_unix_time']
					fvl_wall_clock_time.append(request_ts)

		fvl_wall_clock_time = sorted(fvl_wall_clock_time)
		fvl_wall_clock_time = [ "{:.2f}".format((x - fvl_wall_clock_time[0])/1000) for x in fvl_wall_clock_time]
		return fvl_wall_clock_time


	@classmethod
	def get_community_model_federation_losses_by_wall_clock_time(cls, policy_collection):
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
		fvl_wall_clock_time = []
		for fed_round in sorted_fedrounds_ids:
			metis_grpc_evaluator_metadata = policy_collection[fed_round]['metis_grpc_evaluator_metadata']
			if 'evaluation_requests' in metis_grpc_evaluator_metadata:
				federation_loss_requests = metis_grpc_evaluator_metadata['evaluation_requests']
			else:
				federation_loss_requests = metis_grpc_evaluator_metadata['metis_grpc_evaluator_metadata']['evaluation_requests']
			for request in federation_loss_requests:
				if "is_community_model" in request and request["is_community_model"] is True:
					request_ts = request['request_unix_time']
					fvl_wall_clock_time.append(request_ts)

		fvl_wall_clock_time = sorted(fvl_wall_clock_time)
		fvl_wall_clock_time = [ "{:.2f}".format((x - fvl_wall_clock_time[0])/1000) for x in fvl_wall_clock_time]
		return fvl_wall_clock_time


	@classmethod
	def get_all_federation_losses_by_wall_clock_time(cls, policy_collection):
		sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
		fvl_wall_clock_time = []
		for fed_round in sorted_fedrounds_ids:
			metis_grpc_evaluator_metadata = policy_collection[fed_round]['metis_grpc_evaluator_metadata']
			if 'evaluation_requests' in metis_grpc_evaluator_metadata:
				federation_loss_requests = metis_grpc_evaluator_metadata['evaluation_requests']
			else:
				federation_loss_requests = metis_grpc_evaluator_metadata['metis_grpc_evaluator_metadata']['evaluation_requests']
			for request in federation_loss_requests:
				request_ts = request['request_unix_time']
				fvl_wall_clock_time.append(request_ts)

		fvl_wall_clock_time = sorted(fvl_wall_clock_time)
		fvl_wall_clock_time = [ "{:.2f}".format((x - fvl_wall_clock_time[0])/1000) for x in fvl_wall_clock_time]
		return fvl_wall_clock_time


	@classmethod
	def compute_wall_clock_time_with_test_score_across_federation(cls, policy_collection, metric, synchronous_policy=True, delay=60):
		wall_time_and_score = []
		total_wall_clock_time = 0
		if synchronous_policy:
			sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
			for fedround_id in sorted_fedrounds_ids:
				fedround_wall_clock_time = np.divide(policy_collection[fedround_id]["fedround_execution_time_ms"], 1000)
				# fedround_wall_clock_time += delay
				# fedround_wall_clock_time -= 300
				# fedround_wall_clock_time += 60
				total_wall_clock_time += fedround_wall_clock_time
				fedround_evaluation = policy_collection[fedround_id]["fedround_evaluation"]["evaluation_results"][metric]
				wall_time_and_score.append((total_wall_clock_time, fedround_evaluation))
		else:
			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				min_training_init_time = float("inf")
				for host in hosts_results:
					min_training_init_time = min(min_training_init_time, int(hosts_results[host]['compute_init_unix_time']))
				for host in hosts_results:
					test_set_evaluations = hosts_results[host]['test_set_evaluations']
					for test_set_evaluation in test_set_evaluations:
						if test_set_evaluation['is_after_community_update'] is True:
							evaluation_score = test_set_evaluation['evaluation_results'][metric]
							evaluation_ts = test_set_evaluation['unix_ms']
							wall_time_and_score.append((evaluation_ts, evaluation_score))
				wall_time_and_score = sorted(wall_time_and_score, key=lambda x: x[0])
				wall_time_and_score = [((x[0] - min_training_init_time)/1000, x[1]) for x in wall_time_and_score]

		return wall_time_and_score


	@classmethod
	def compute_wall_clock_time_with_federation_set_accuracy_score(cls, policy_collection):
		wall_time_and_score = []
		for fed_round in policy_collection:
			metis_grpc_evaluator_metadata = policy_collection[fed_round]['metis_grpc_evaluator_metadata']
			federation_loss_requests = metis_grpc_evaluator_metadata['evaluation_requests']
			min_training_init_time = float("inf")
			for request in federation_loss_requests:
				if request['is_community_model'] is True:
					request_unix_ts = request['request_unix_time']
					norm_key = 'validation_batches_num' if 'validation_batches_num' in request['learner_federated_validation'][0] else 'validation_dataset_size'
					norm_factor = sum([learner_request[norm_key] for learner_request in request['learner_federated_validation']])
					request_score = sum([(learner_request[norm_key] * learner_request['validation_accuracy']) / norm_factor
										 for learner_request in request['learner_federated_validation']])
					wall_time_and_score.append((request_unix_ts, request_score))
					min_training_init_time = min(min_training_init_time, request_unix_ts)
			wall_time_and_score = sorted(wall_time_and_score, key=lambda x: x[0])
			wall_time_and_score = [((x[0] - min_training_init_time)/1000, x[1]) for x in wall_time_and_score]

		return wall_time_and_score



	@classmethod
	def compute_epochs_processing_time_across_federation(cls, policy_collection, metric, synchronous_policy=True, max_execution_seconds=None, delay=0):

		total_processing_time = 0
		total_execution_time = 0
		all_hosts_proc_score = list()
		if max_execution_seconds is None:
			max_execution_seconds = float("inf")
		if synchronous_policy:
			sorted_fedrounds_ids = LogsProcessingUtil.sort_alphanumeric(policy_collection, reverse=False)
			for fed_round in sorted_fedrounds_ids:
				hosts_results = policy_collection[fed_round]['hosts_results']
				fed_round_processing_time = 0
				for host in hosts_results:
					host_epochs_times = hosts_results[host]['epochs_exec_times_ms']
					host_processing_time = np.sum(host_epochs_times)
					host_processing_time /= 1000
					fed_round_processing_time = max(host_processing_time, fed_round_processing_time)
					# fed_round_processing_time += host_processing_time # for cumulative time
				total_processing_time += fed_round_processing_time
				current_processing_time_point = all_hosts_proc_score[-1][0] + fed_round_processing_time if len(all_hosts_proc_score) > 0 else fed_round_processing_time
				fed_round_accuracy = policy_collection[fed_round]['fedround_evaluation']['evaluation_results'][metric]

				all_hosts_proc_score.append((current_processing_time_point, fed_round_accuracy))

				# fedround_wall_clock_time = np.divide(policy_collection[fed_round]["fedround_execution_time_ms"], 1000)
				# fedround_wall_clock_time += delay
				# total_execution_time += fedround_wall_clock_time
				# if total_execution_time <= max_execution_seconds:
				# 	all_hosts_proc_score.append((current_processing_time_point, fed_round_accuracy))
		else:
			all_hosts_ts_proc_score = []
			all_hosts_ts_proc_score_dict = defaultdict(list)
			for fed_round in policy_collection:
				hosts_results = policy_collection[fed_round]['hosts_results']
				hosts_total_processing_time = 0
				for host in hosts_results:
					training_start_time = int(hosts_results[host]['compute_init_unix_time'])
					host_ts_proc_score = []
					host_epochs_times_ms = hosts_results[host]['epochs_exec_times_ms']
					host_epochs_times_sec = [x/1000 for x in host_epochs_times_ms]
					host_total_processing_time = sum(host_epochs_times_sec)
					hosts_total_processing_time += host_total_processing_time

					test_set_evaluations = hosts_results[host]['test_set_evaluations']
					# Left and Right indexes associate with the slice of epoch times that we need
					left_idx = 0
					right_idx = 0
					previous_timestamp = -1
					current_timestamp = -1
					for eval_idx, evaluation in enumerate(test_set_evaluations):
						score_value = float(evaluation['evaluation_results'][metric])
						current_duration_sec = (current_timestamp - training_start_time) / 1000
						if current_duration_sec < max_execution_seconds:
							total_processing_time += current_duration_sec
							if evaluation['is_after_community_update'] is False:
								right_idx += 1
								previous_timestamp = current_timestamp
								current_timestamp = evaluation['unix_ms']
							if evaluation['is_after_community_update'] is True:
								processing_time = sum(host_epochs_times_sec[left_idx:right_idx])
								host_ts_proc_score.append((previous_timestamp, host, processing_time, score_value))
								all_hosts_ts_proc_score_dict[host].append((previous_timestamp, processing_time, score_value))
								left_idx = right_idx
					all_hosts_ts_proc_score.extend(host_ts_proc_score)


			all_hosts_ts_proc_score_sorted = sorted(all_hosts_ts_proc_score, key=lambda x: x[0])
			previous_ts = all_hosts_ts_proc_score_sorted[0][0]
			proc_with_scores = []
			for vals in  all_hosts_ts_proc_score_sorted:
				if len(proc_with_scores) == 0:
					proc_with_scores.append((int((vals[0]-previous_ts)/1000), vals[3]))
				else:
					proc_with_scores.append((proc_with_scores[-1][0] + int((vals[0] - previous_ts) / 1000), vals[3]))
				previous_ts = vals[0]

			all_hosts_proc_score = proc_with_scores

			# periods_with_scores = []
			# previous_updated_learner = None
			# rolling_learners_proc_time = dict()
			# for k in all_hosts_ts_proc_score_dict.keys():
			# 	rolling_learners_proc_time[k] = 0
			# for val in all_hosts_ts_proc_score_sorted:
			# 	current_timestamp = val[0]
			# 	current_updating_learner = val[1]
			# 	current_proc_time = val[2]
			# 	current_score_value = val[3]
			# 	if previous_updated_learner is None:
			# 		period_size = rolling_learners_proc_time[current_updating_learner] + current_proc_time
			# 	else:
			# 		period_size = rolling_learners_proc_time[current_updating_learner] - rolling_learners_proc_time[previous_updated_learner] + current_proc_time
			# 	rolling_learners_proc_time[current_updating_learner] = rolling_learners_proc_time[current_updating_learner] + current_proc_time
			# 	previous_updated_learner = current_updating_learner
			# 	periods_with_scores.append((period_size, current_score_value))
			#
			# final_periods_with_scores = [periods_with_scores[0]]
			# for p_score in periods_with_scores[1:]:
			# 	final_periods_with_scores.append((final_periods_with_scores[-1][0] + p_score[0], p_score[1]))
			# all_hosts_proc_score = final_periods_with_scores


			# # THIS IS FOR CUMULATIVE PROCESSING TIME
			# # Sort list with hosts scores by increasing unix timestamp
			# all_hosts_ts_proc_score = sorted(all_hosts_ts_proc_score, key=lambda x: x[0])
			# # Create processing buckets
			# window_size_ms = 60000
			# init_time = all_hosts_ts_proc_score[0][0]
			# all_hosts_proc_score_buckets = []
			# current_processing_time = all_hosts_ts_proc_score[0][1]
			# current_processing_time_score = all_hosts_ts_proc_score[0][2]
			# for tpl in all_hosts_ts_proc_score[1:]:
			# 	end_time = tpl[0]
			# 	if (end_time - init_time) < window_size_ms:
			# 		current_processing_time += tpl[1]
			# 		current_processing_time_score = max(tpl[2], current_processing_time_score)
			# 	else:
			# 		all_hosts_proc_score_buckets.append((current_processing_time, current_processing_time_score))
			# 		init_time = end_time
			# 		current_processing_time = tpl[1]
			# 		current_processing_time_score = tpl[2]
			#
			# all_hosts_proc_score = [all_hosts_proc_score_buckets[0]]
			# for bucket in all_hosts_proc_score_buckets:
			# 	current_proc_time = bucket[0] + all_hosts_proc_score[-1][0]
			# 	score_value = bucket[1]
			# 	all_hosts_proc_score.append((current_proc_time, score_value))

			# # THIS IS FOR MAXIMUM PROCESSING TIME BETWEEN UPDATE REQUESTS
			# window_size = 60000 # ms
			# hosts_window_based_max_proc_time_score = defaultdict(list)
			# for host, values in all_hosts_ts_proc_score_dict.items():
			# 	values = sorted(values, key=lambda x:x[0])
			# 	ts_0 = values[0][0]
			# 	host_proc_time = values[0][1]
			# 	host_score = values[0][2]
			# 	target_time = ts_0 + window_size
			# 	for val in values[1:]:
			# 		if val[0] > target_time:
			# 			hosts_window_based_max_proc_time_score[host].append((host_proc_time, host_score))
			# 			# target_time = val[0] + window_size
			# 			target_time += window_size
			# 			host_proc_time = val[1]
			# 			host_score = val[2]
			# 		else:
			# 			host_proc_time += val[1]
			# 			host_score = max(host_score, val[2])
			#
			# all_hosts_proc_score = []
			# maximum_windows_number = max([len(val) for val in hosts_window_based_max_proc_time_score.values()])
			# current_proc_time = 0
			# for idx in range(maximum_windows_number):
			# 	current_window_proc_score = [-np.inf, -np.inf]
			# 	for host, values in hosts_window_based_max_proc_time_score.items():
			# 		if len(values) <= idx:
			# 			continue
			# 		else:
			# 			host_proc = values[idx][0]
			# 			host_score = values[idx][1]
			# 			current_window_proc_score = [max(current_window_proc_score[0], host_proc),
			# 										 max(current_window_proc_score[1], host_score)]
			# 	current_proc_time = current_proc_time + current_window_proc_score[0]
			# 	all_hosts_proc_score.append((current_proc_time, current_window_proc_score[1]))

		return total_processing_time, all_hosts_proc_score