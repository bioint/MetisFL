from utils.logging.logs_processing_ops import LogsProcessingUtil
from matplotlib.patches import Rectangle
from collections import defaultdict, Counter

import csv
import random
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


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
	age_buckets = [(39, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
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

		ax[idx].set_xlim(40, 100)
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
		ax[idx].text(0.02, 0.93, textstr, transform=ax[idx].transAxes, fontsize=17.5,
				verticalalignment='top', bbox=props)


		# fig.text(0.07, 0.55, 'Probability', ha='center', va='center', rotation='vertical', fontsize=12)
		fig.text(0.01, 0.55, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=18)

	return fig


def generate_stratified_validation_dataset(train_path, idx_file, validation_dataset_size_pct=0.05):
	train_data = pd.read_csv(train_path)
	train_data["age_bin"] = train_data["age_at_scan"].astype(int)
	training_num_examples = len(train_data.index)
	validation_num_examples = np.rint(np.multiply(training_num_examples, validation_dataset_size_pct))

	step = 5
	age_buckets = [(x, x + step) for x in range(39, 100, step)]
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
			x = x[x["9dof_2mm_vol"] != elem["9dof_2mm_vol"]]
			train_data = train_data[train_data["9dof_2mm_vol"] != elem["9dof_2mm_vol"]]
			validation_data.append(elem)

	output_directory = os.path.dirname(train_path)
	new_train_data = pd.DataFrame(train_data)
	new_validation_data = pd.DataFrame(validation_data)

	new_train_data.to_csv(output_directory + "/v_train_{}.csv".format(idx_file), index=False)
	new_validation_data.to_csv(output_directory + "/v_valid_{}.csv".format(idx_file), index=False)

	print(age_buckets)
	print("Expected Train/Valid Sizes: ", training_num_examples - validation_num_examples, validation_num_examples)
	print("Actual Train/Valid Sizes: ", len(new_train_data), len(new_validation_data))


def generate_learners_training_and_test_dataset_distribution(train_path):
	subj_accelerated_scans = defaultdict(list)
	subj_non_accelerated_scans = defaultdict(list)
	with open(train_path) as fin:
		csv_reader = csv.DictReader(fin)
		for line in csv_reader:
			if "Accel" == line["type"]:
				subj_accelerated_scans[line["subject_id"]].append(line)
			else:
				subj_non_accelerated_scans[line["subject_id"]].append(line)

	total_accelerated = len([scan for all_scans in subj_accelerated_scans.values() for scan in all_scans])
	total_non_accelerated = len([scan for all_scans in subj_non_accelerated_scans.values() for scan in all_scans])
	print("Number of accelerated and non-aacelerated records: ", total_accelerated, total_non_accelerated)

	# The rational is to reserve 20% of samples for test and split the rest 80% to two parts of 40%.
	# First two partitions will get the accelerated scans, while the last two partitions will get the non-accelerated.
	num_partitions = 4
	partitions_data = defaultdict(list)
	partitions_sizes = [int(np.multiply(0.4, total_accelerated)), int(np.multiply(0.4, total_accelerated)),
						int(np.multiply(0.4, total_non_accelerated)), int(np.multiply(0.4, total_non_accelerated))]
	print("Expected partitions sizes: ", partitions_sizes)

	for partition_id in range(num_partitions):
		partition_size = partitions_sizes[partition_id]
		current_size = 0
		if partition_id <= 1:
			while current_size < partition_size:
				if subj_accelerated_scans:
					current_scans = subj_accelerated_scans.popitem()[1]
					partitions_data[partition_id].extend(current_scans)
					current_size += len(current_scans)
				else:
					break
		else:
			while current_size < partition_size:
				if subj_non_accelerated_scans:
					current_scans = subj_non_accelerated_scans.popitem()[1]
					partitions_data[partition_id].extend(current_scans)
					current_size += len(current_scans)
				else:
					break
		print(partition_id, current_size)

	test_dataset = subj_accelerated_scans
	for k, v in subj_non_accelerated_scans.items():
		if k in test_dataset:
			test_dataset[k].extend(v)
		else:
			test_dataset[k] = v

	for partition_id, partition_data in partitions_data.items():
		for record in partition_data:
			if record['subject_id'] in test_dataset:
				subject_id = record['subject_id']
				subject_id_scans = test_dataset.pop(subject_id)
				if subject_id_scans[0]['type'] == "Accel" and partition_id <= 1:
					partition_data.extend(subject_id_scans)
				if subject_id_scans[0]['type'] == "Non-Accel" and partition_id > 1:
					partition_data.extend(subject_id_scans)

	dups = 0
	for partition_id, partition_data in partitions_data.items():
		for record in partition_data:
			if record['subject_id'] in test_dataset:
				dups += 1
	print("Duplicate subjects in test and training partitions: ", dups)

	test_dataset_scans = [(scan['subject_id'],scan['age_at_scan'],scan['9dof_2mm_vol']) for subject_scans in test_dataset.values() for scan in subject_scans]
	pd.DataFrame(test_dataset_scans).to_csv("generated_distributions/test.csv", index=False)


	federation_data_distribution = defaultdict(dict)
	for partition_id, val in partitions_data.items():
		pidx = str(int(partition_id) + 1)
		pd.DataFrame(val).to_csv("generated_distributions/train_{}.csv".format(pidx), index=False)
		age_data = [ float(x['age_at_scan']) for x in val]
		print(age_data)
		print("Partition ID: ", pidx)
		print("Partition Size: ", len(age_data))
		print("Partition Data: ", age_data)
		print("Mean: {}, STD: {}".format(np.mean(age_data), np.std(age_data)))
		federation_data_distribution[pidx]['train_stats'] = dict()
		federation_data_distribution[pidx]['train_stats']['dataset_size'] = len(age_data)
		federation_data_distribution[pidx]['train_stats']['dataset_values'] = age_data

	return partitions_data, federation_data_distribution


if __name__=="__main__":
	TRAIN_PATH = "../ADNI_collated_4k.csv"
	node_to_data, federation_data_distribution = generate_learners_training_and_test_dataset_distribution(TRAIN_PATH)

	fig = plot_learners_age_buckets(federation_data_distribution)
	file_out = "generated_distributions/AgeBuckets.png"
	fig.savefig(file_out, bbox_inches='tight')

	fig = plot_learners_age_distributions(federation_data_distribution)
	file_out = "generated_distributions/AgeDistributions.png"
	fig.savefig(file_out, bbox_inches='tight')

	train_files = glob.glob("generated_distributions/train*.csv")
	train_files = sorted(train_files) # need to sort them to go with increasing train_id order.
	for file in train_files:
		fidx = int(file.split('_')[-1].split('.csv')[0])
		print(fidx)
		generate_stratified_validation_dataset(file, idx_file=fidx)
		print("\n\n")

