import random
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from collections import defaultdict
from matplotlib.patches import Rectangle
from utils.logging.logs_processing_ops import LogsProcessingUtil


np.random.seed(seed=1990)
random.seed(a=1990)


def plot_learners_age_buckets(federation_data_distribution):

	fig, ax = plt.subplots(nrows=1, ncols=1)
	# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

	ax.grid(True)
	ax.set_xlabel("Learners", fontsize=24)
	ax.set_ylabel("#Examples", fontsize=24)
	# ax.set_ylim(0.45, 0.85)  # Cifar10
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(24)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(24)

	fig.subplots_adjust(bottom=0.22)

	# Intervals are open left - closed right. Example: (39, 50], (50, 60]
	age_buckets = [(39, 50), (50, 60), (60, 70), (70, 80)]
	age_buckets_num = len(age_buckets)
	age_buckets_indexes = range(0, age_buckets_num)
	age_buckets_interval_index = pd.IntervalIndex.from_tuples(age_buckets)
	learners_ids_sorted_by_partition_size = sorted(
		federation_data_distribution,
		key=lambda k: (federation_data_distribution[k]['train_stats']['dataset_size'], k),
		reverse=True)
	learners_num = len(learners_ids_sorted_by_partition_size)
	learners_ids_by_idx = range(0, learners_num)
	learners_age_buckets = dict()

	for learner_id in learners_ids_sorted_by_partition_size:
		train_stats = federation_data_distribution[learner_id]['train_stats']
		train_dataset_values = train_stats['dataset_values']
		train_dataset_values_sorted = sorted(train_dataset_values)
		train_dataset_size = train_stats['dataset_size']

		learner_buckets_df = pd.cut(train_dataset_values_sorted, bins=age_buckets_interval_index)
		learners_age_buckets[learner_id] = list(learner_buckets_df.value_counts())
		print(learner_id, learners_age_buckets[learner_id])


	ind = np.arange(learners_num)  # the x locations for the groups
	width = 0.35  # the width of the bars: can also be len(x) sequence
	hist_colors = LogsProcessingUtil.get_colors_map(n=age_buckets_num, name='gnuplot')
	# hist_colors = LogsProcessingUtil.get_colors_map(n=age_buckets_num, name='viridis')

	bars_to_plot = []
	for age_bucket_idx in age_buckets_indexes:
		age_bucket_idx_data = []
		for learner_id in learners_age_buckets.keys():
			age_bucket_idx_examples_num = learners_age_buckets[learner_id][age_bucket_idx]
			age_bucket_idx_data.append(age_bucket_idx_examples_num)
		bars_to_plot.append(age_bucket_idx_data)
	print(bars_to_plot)

	plot_bars_legends = []
	for bar_idx, bar_to_plot in enumerate(bars_to_plot):
		bar_bottom = [sum(x) for x in zip(*bars_to_plot[:bar_idx])]
		if len(bar_bottom) == 0:
			bar_bottom = 0
		plot_bar = ax.bar(ind, bar_to_plot, width, bottom=bar_bottom, color=hist_colors(bar_idx))
		plot_bars_legends.append(plot_bar)


	ax.set_xticks(ind)
	ax.set_xticklabels([ "L{}".format(idx+1) for idx in learners_ids_by_idx], rotation=30, ha="right", fontsize=24)
	empty_box = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
	age_buckets = [ "({},{}]".format(bucket[0], bucket[1]) for bucket in age_buckets]
	# plot_bars_legends.insert(0, empty_box)
	# age_buckets.insert(0, "Age Buckets")
	# legend = ax.legend(plot_bars_legends, age_buckets, framealpha=0.0, prop={'size': 24}, loc='center left',
	# 				   bbox_to_anchor=(1, 0.5))
	# We perform the reverse so that the age buckets are in descending order.
	plot_bars_legends.insert(len(age_buckets), empty_box)
	age_buckets.insert(len(age_buckets), "Age Buckets")
	legend = ax.legend(reversed(plot_bars_legends), reversed(age_buckets), framealpha=0.0, prop={'size': 24}, loc='center left',
					   bbox_to_anchor=(1, 0.5))


	return fig


def plot_learners_age_distributions(federation_data_distribution):

	learners_ids_sorted_by_partition_size = sorted(
		federation_data_distribution,
		key=lambda k: (federation_data_distribution[k]['train_stats']['dataset_size'], k),
		reverse=True)
	learners_num = len(learners_ids_sorted_by_partition_size)


	fig, ax = plt.subplots(nrows=learners_num, ncols=1, figsize=(8,12))
	color_map = LogsProcessingUtil.get_colors_map(n=int(learners_num*1.3), name='OrRd_r')

	# ax[0].set_title('Learners Training Dataset Distribution', fontsize=12)
	ax[0].set_title('Learners Training Dataset Distribution', fontsize=18)
	ax[learners_num-1].set_xlabel(r'$\bf{Age}$', fontsize=18)
	# ax.set_xlabel("Age", fontsize=18)
	# ax.set_ylabel("Density", fontsize=18)
	# # ax.set_ylim(0.45, 0.85)  # Cifar10
	for axis in ax:
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(16)
		for tick in axis.yaxis.get_major_ticks():
			tick.label.set_fontsize(16)

	fig.subplots_adjust(bottom=0.22)


	learners_ages = dict()
	for idx, learner_id in enumerate(learners_ids_sorted_by_partition_size):
		train_stats = federation_data_distribution[learner_id]['train_stats']
		train_dataset_values = train_stats['dataset_values']
		train_dataset_values_sorted = sorted(train_dataset_values)
		train_dataset_size = train_stats['dataset_size']

		learners_ages[learner_id] = train_dataset_values_sorted

		mu = np.mean(train_dataset_values_sorted)
		variance = np.var(train_dataset_values_sorted)
		sigma = np.std(train_dataset_values_sorted)
		distrib = stats.norm.pdf(train_dataset_values_sorted, mu, sigma)

		if idx < learners_num-1:
			ax[idx].tick_params(
				axis='x',  # changes apply to the x-axis
				which='both',  # both major and minor ticks are affected
				bottom=False,  # ticks along the bottom edge are off
				top=False,  # ticks along the top edge are off
				labelbottom=False)  # labels along the bottom edge are off

		ax[idx].set_xlim(40, 85)
		# ax[idx].set_ylim(0.0, 0.1)

		ax[idx].grid(True)
		# ax[idx].plot(train_dataset_values_sorted, distrib, color=color_map(idx))
		# ax[idx].fill_between(train_dataset_values_sorted, distrib, distrib, facecolor=color_map(idx), alpha='1.0')
		# ax[idx].legend([np.round(mu,decimals=3), np.round(sigma, decimals=3)], frameon=False)

		num_bins = 50
		# the histogram of the data
		n, bins, patches = ax[idx].hist(train_dataset_values_sorted, num_bins, density=1, color=color_map(idx),
										alpha=0.75)
		# add a 'best fit' line
		distrib = stats.norm.pdf(bins, mu, sigma)
		l = ax[idx].plot(bins, distrib, 'r--', linewidth=1)

		textstr = '\n'.join((r'$\mathrm{\bf{'+str("L{}".format(idx+1))+'}}$',
							 r'$\mu=%.2f$' % (np.round(mu, decimals=2),),
							 r'$\sigma=%.2f$' % (np.round(sigma, decimals=2),)))

		# text box properties
		props = dict(boxstyle='round', facecolor='white', alpha=1)
		# place a text box in upper left in axes coords
		# ax[idx].text(0.89, 0.9, textstr, transform=ax[idx].transAxes, fontsize=14,
		# 		verticalalignment='top', bbox=props)
		ax[idx].text(0.95, 1.05, textstr, transform=ax[idx].transAxes, fontsize=17.5,
				verticalalignment='top', bbox=props)


		# fig.text(0.07, 0.55, 'Probability', ha='center', va='center', rotation='vertical', fontsize=12)
		fig.text(0.01, 0.55, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=18)

	return fig


def get_partitions_sizes(skewness_factor, partitions, num_examples, descending=True):
	# Find smallest bin size based on the factor difference between two consecutive partitions.
	# CAUTION pidx starts from 0, thus +1 (need it for factorized bin sizes)
	factorized_bin_sizes = [np.power(skewness_factor, pidx + 1) for pidx in range(partitions)]
	bin_size_factor = np.floor(num_examples / sum(factorized_bin_sizes))

	# Find actual bin sizes using the smallest bin size (bin_size_factor).
	partitions_sizes = [np.floor(np.power(skewness_factor, pidx + 1) * bin_size_factor) for pidx in range(partitions)]
	total_partitions_size = sum(partitions_sizes)
	if total_partitions_size < num_examples:
		# If partition sizes are not summing up to the total
		# then start assigning remaining examples from the largest
		# bin to the smallest (thus the reversed call).
		new_partitions_sizes = list(reversed(partitions_sizes))
		remaining_examples = num_examples - total_partitions_size
		for idx, psize in enumerate(new_partitions_sizes):
			# new data size depends on factor
			increment = np.ceil(remaining_examples / skewness_factor)
			new_partitions_sizes[idx] = psize + increment
			remaining_examples -= increment
		# Bring the partitions into ascending size order (smallest->largest)
		partitions_sizes = list(new_partitions_sizes)

	partitions_sizes = sorted(partitions_sizes, reverse=descending)

	return partitions_sizes


def generate_learners_training_dataset_distribution(train_path):
	data = pd.read_csv(train_path)
	data["age_bin"] = data["age_at_scan"].astype(int)
	num_examples = len(data.index)

	# SIZES = get_partitions_sizes(skewness_factor=1.35, partitions=8, num_examples=num_examples, descending=True)
	SIZES = get_partitions_sizes(skewness_factor=1.35, partitions=8, num_examples=num_examples, descending=True)
	# SIZES = [900, 900, 900, 900, 900, 900, 900, 900]
	# SIZES = [200, 200, 200, 200, 200, 200, 200, 200]
	# MEANS = [80, 70, 60, 50, 40, 30, 80, 70]
	MEANS = [90, 70, 50, 30, 90, 70, 50, 30]
	# MEANS = [80, 70, 60, 50, 80, 70, 60, 50]
	# MEANS = [80, 80, 70, 70, 60, 60, 50, 50]
	STD = 10

	# Map site idx to list of dataframes
	node_to_data = defaultdict(list)
	keep_running = True

	# Pass 1
	# Stop when dataset is empty or when all sites are at capacity
	while keep_running and len(data) > 0:

		keep_running = False  # Stop when all sites reach capacity
		for i in range(len(MEANS)):

			if len(data) == 0:
				break

			if len(node_to_data[str(i)]) < SIZES[i]:
				keep_running = True
				# Sample from site (Keep re-sampling till valid)
				while True:
					sample = int(np.random.normal(loc=MEANS[i], scale=STD))

					if len(data[data["age_bin"] == sample]) != 0:
						x = data[data["age_bin"] == sample].iloc[0]
						# Add subj to site set
						node_to_data[str(i)].append(x)
						# Remove subj from dataset
						data = data[data["eid"] != x["eid"]]
						break
		print(len(data))

	for i in range(len(SIZES)):
		print(len(node_to_data[str(i)]), SIZES[i])

	# Pass 2
	# Fill in remaining examples based on descending bin sizes
	# If still remaining data, add them arbitrarily
	# uniformly across sites
	# while len(data) > 0:
	#     for i in range(len(SIZES)):
	#         if len(data) > 0:
	#             x = data.iloc[0]
	#             # Add subj to site set
	#             node_to_data[str(i)].append(x)
	#             data = data[data["eid"] != x["eid"]]

	federation_data_distribution = defaultdict(dict)
	for sidx in node_to_data:
		fidx = str(int(sidx) + 1)
		pd.DataFrame(node_to_data[sidx]).to_csv("generated_distributions/train_{}.csv".format(fidx), index=False)
		partition_data = [x['age_at_scan'] for x in node_to_data[sidx]]
		print("Partition ID: ", fidx)
		print("Partition Size: ", len(partition_data))
		print("Partition Data: ", partition_data)
		print("Mean: {}, STD: {}".format(np.mean(partition_data), np.std(partition_data)))
		federation_data_distribution[fidx]['train_stats'] = dict()
		federation_data_distribution[fidx]['train_stats']['dataset_size'] = len(partition_data)
		federation_data_distribution[fidx]['train_stats']['dataset_values'] = partition_data

	return node_to_data, federation_data_distribution


def generate_stratified_validation_dataset(train_path, idx_file, validation_dataset_size_pct=0.05):
	train_data = pd.read_csv(train_path)
	train_data["age_bin"] = train_data["age_at_scan"].astype(int)
	training_num_examples = len(train_data.index)
	validation_num_examples = np.rint(np.multiply(training_num_examples, validation_dataset_size_pct))

	# age_buckets = [(39, 45), (45, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 80)]
	step = 5
	age_buckets = [(x, x + step) for x in range(39, 80, step)]
	age_buckets_num = len(age_buckets)
	age_buckets_indexes = range(0, age_buckets_num)
	age_buckets_interval_index = pd.IntervalIndex.from_tuples(age_buckets)

	buckets_df = pd.cut(train_data['age_bin'].values, bins=age_buckets_interval_index)
	buckets_counts_df = buckets_df.value_counts()

	validation_data = list()
	for age_bucket_idx in age_buckets_indexes:
		min_bucket_val, max_bucket_val = age_buckets[age_bucket_idx][0], age_buckets[age_bucket_idx][1]
		bucket_size = buckets_counts_df[age_bucket_idx]
		bucket_proportion = np.divide(bucket_size, training_num_examples)
		# num_validation_elements = np.int(np.floor(np.multiply(validation_num_examples, proportion)))
		num_validation_elements = np.int(np.rint(np.multiply(validation_num_examples, bucket_proportion)))
		x = train_data[(train_data["age_bin"] > min_bucket_val) & (train_data['age_bin'] <= max_bucket_val)]
		for i in range(num_validation_elements):
			elem = x.iloc[0]
			x = x[x["eid"] != elem["eid"]]
			train_data = train_data[train_data["eid"] != elem["eid"]]
			validation_data.append(elem)

	pd.DataFrame(train_data).to_csv("train_{}.csv".format(idx_file), index=False)
	pd.DataFrame(validation_data).to_csv("valid_{}.csv".format(idx_file), index=False)
	print("\n\n")
	print(age_buckets)
	print(training_num_examples - validation_num_examples, validation_num_examples)
	print(len(train_data.index), len(validation_data))


if __name__ == "__main__":
	TRAIN_PATH = "../centralized/train.csv"
	node_to_data, federation_data_distribution = generate_learners_training_dataset_distribution(TRAIN_PATH)

	fig = plot_learners_age_buckets(federation_data_distribution)
	file_out = "generated_distributions/AgeBuckets.png"
	fig.savefig(file_out, bbox_inches='tight')

	fig = plot_learners_age_distributions(federation_data_distribution)
	file_out = "generated_distributions/AgeDistributions.png"
	fig.savefig(file_out, bbox_inches='tight')


	# plt.show()

# train_files = glob.glob("../uniform_datasize_noniid_x8clients/small_consortia/without_validation/train*.csv")
# for fidx, file in enumerate(train_files, start=1):
#     generate_stratified_validation_dataset(file, idx_file=fidx)
