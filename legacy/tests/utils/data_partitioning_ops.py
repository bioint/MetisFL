from matplotlib.backends.backend_pdf import PdfPages
from utils.objectdetection.imgdata_client import MetisDBSession
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random

input_arr = np.array([ [0,1],
					   [0,2],
					   [1,1],
					   [1,2],
					   [2,1],
					   [2,2],
					   [3,1],
					   [3,2],
					   [4,1],
					   [4,2],
					   [5,1],
					   [5,2],
					   [6,1],
					   [6,2],
					   [7,1],
					   [7,2],
					   [8,1],
					   [8,2],
					   [9,1],
					   [9,2]])

output_arr = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])


class PartitionDataset(object):

	def __init__(self, input, output, partition_size, partition_classes_stats=None):
		self.input = input
		self.output = output
		self.partition_size = partition_size
		self.partition_classes = np.unique(self.output)
		self.partition_classes_stats = partition_classes_stats
		if partition_classes_stats is None:
			self.compute_partition_distribution()

	def __str__(self):
		data_distribution_str = [str(self.partition_classes_stats[x]) for x in self.partition_classes_stats]
		return "Input Data Shape: {}, " \
			   "Output Data Shape: {}, " \
			   "Partition Size: {}, " \
			   "Partition Classes: {}, " \
			   "Partition Data Distribution: {}".format(self.input.shape,
														self.output.shape,
														self.partition_size,
														self.partition_classes,
														data_distribution_str)

	def compute_partition_distribution(self):
		percentages = Counter(self.output)
		for key in percentages.keys():
			percentages[key] = PartitionDatasetClassStats(class_id=key,
														  class_number_of_examples=percentages[key],
														  class_examples_percentage=round(percentages[key] / self.partition_size, 2))
		sorted(percentages)
		self.partition_classes_stats = percentages


class PartitionDatasetClassStats(object):
	def __init__(self, class_id, class_number_of_examples, class_examples_percentage):
		self.class_id = class_id
		self.class_number_of_examples = class_number_of_examples
		self.class_examples_percentage = class_examples_percentage

	def __str__(self):
		return "<class id: {}, number of examples: {}, examples percentage: {}>".format(self.class_id,
																						self.class_number_of_examples,
																						self.class_examples_percentage)

# partitioned_data = DataPartitioningUtil.iid_partitioning(input_data=input_arr, output_classes=output_arr, partitions=10)
# for pid in partitioned_data:
# 	print(pid, partitioned_data[pid].input, partitioned_data[pid].output)
# partitioned_data = DataPartitioningUtil.balanced_class_partitioning(input_data=input_arr, output_classes=output_arr, partitions=10, classes_per_partition=2)
# for pid in partitioned_data:
# 	print(pid, partitioned_data[pid].input, partitioned_data[pid].output)

# Unbalanced Data Size Per Partition Formula:
# smallest_partition_size = Σ(partition_diff_factor ^ partition_num) / #num_examples
# any_partition_size = partition_diff_factor ^ partition_num * smallest_partition_size

def unbalanced_data_split_test():
	data_distribution = [[5000,'A'], [5000, 'B'], [5000, 'C'], [5000, 'D'], [5000, 'E'],
						 [5000, 'F'], [5000, 'G'], [5000, 'H'], [5000, 'I'], [5000, 'J']]
	num_examples = sum([x[0] for x in data_distribution])
	num_classes = 10
	num_classes_per_partition = 5
	number_of_partitions = 10
	diff_factor_between_partitions = 1.6
	partition_factor_sizes = [np.power(diff_factor_between_partitions, idx) for idx in range(number_of_partitions)]
	smallest_partition_data_size = np.floor(num_examples / sum(partition_factor_sizes))
	total_data = 0

	partitions_data_size = []
	for p in range(number_of_partitions):
		partition_data_size = np.power(diff_factor_between_partitions, p) * smallest_partition_data_size
		partitions_data_size.append(int(partition_data_size))
		print(p, partition_data_size)
	print(sum(partitions_data_size))
	print(partitions_data_size)
	print()

	partition_data_assignment = []
	data_distribution = sorted(data_distribution, key=lambda x: x[0])
	for idx, psize  in enumerate(partitions_data_size):
		p_assignment = []
		class_data_for_partition = np.floor(psize/num_classes_per_partition)
		assigned_classes = 0
		for di, x in enumerate(data_distribution):
			class_size = x[0]
			class_label = x[1]
			if psize <= 0 or class_size == 0:
				continue
			else:
				to_insert = min(class_data_for_partition, class_size)
				p_assignment.append([to_insert, class_label])
				data_distribution[di][0] -= to_insert
				psize -= to_insert
				assigned_classes += 1
		partition_data_assignment.append(p_assignment)

	total_data = 0
	partitions_total_data = []
	for p in partition_data_assignment:
		partition_total_data = sum([k[0] for k in p])
		partitions_total_data.append(partition_total_data)
		total_data += partition_total_data
		print(p)
	print(total_data)

	# Add all remaining classes to the learner with the highest number of examples
	for idx, distrib  in enumerate(data_distribution):
		if distrib[0] != 0:
			partitions_total_data[-1] += distrib[0]
			data_distribution[idx][0] = 0

	# Plot the reversed (long tail)
	# plt.bar(x=[x+1 for x in range(number_of_partitions)], height=sorted(partitions_total_data)[::-1])
	bar_y_values = []
	labels = [x[1] for x in data_distribution]
	bar_x_values = [0] * len(labels)
	ind = np.arange(len(labels))    # the x locations for the groups
	width = 0.35
	bars_pointers = []
	for lid in labels:
		lid_bar_x_values = []
		for x in partition_data_assignment[::-1]:
			not_found = True
			for size, label in x:
				if lid == label:
					lid_bar_x_values.append(size)
					not_found = False
			if not_found:
				lid_bar_x_values.append(0)
		bar_pointer = plt.bar(ind, lid_bar_x_values, width, bottom=bar_x_values)
		bars_pointers.append(bar_pointer)
		bar_x_values = [sum(x) for x in zip(bar_x_values, lid_bar_x_values)]
	plt.legend(bars_pointers, labels)

	plt.xticks([x for x in range(number_of_partitions)])
	plt.xlabel("Learner ID")
	plt.ylabel("Data per Learner")
	plt.show()


def unbalanced_data_partitioning(input, output, num_partitions, classes_per_partition, skewness_factor):

	assert isinstance(input, np.ndarray)
	assert isinstance(output, np.ndarray)

	# Sort data according to class id.
	permutation = output.argsort()
	input_data = input[permutation]
	output_classes = output[permutation]

	# Get total number of examples.
	num_examples = np.size(output_classes)
	unique_classes = list(np.unique(output_classes))

	# Find smallest bin size based on the factor difference between two consecutive partitions.
	factorized_bin_sizes = [np.power(skewness_factor, idx) for idx in range(num_partitions)]
	smallest_bin_size = np.floor(num_examples / sum(factorized_bin_sizes))

	# Find actual bin sizes using the smallest bin size.
	partitions_sizes = [np.floor(np.power(skewness_factor, idx) * smallest_bin_size) for idx in range(num_partitions)]
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
		partitions_sizes = list(reversed(new_partitions_sizes))

	partitions_datasets = []
	total_examples_assigned = 0
	# Pass 1: Fill in the partition buckets
	# 	a. Find number of examples of each class per partition
	# 	b. Loop through the classes and assign examples to partitions
	for pid, psize in enumerate(partitions_sizes):

		current_size = 0
		examples_per_class = int(np.ceil(psize / classes_per_partition))
		partition_input = list()
		partition_output = list()

		# Shuffle classes id to add randomness in our partition distribution
		random.shuffle(unique_classes)
		for cid in unique_classes:
			if current_size < psize:
				partition_remaining_slots = psize - current_size
				remaining_assigning_examples = min(partition_remaining_slots, examples_per_class)
				current_class_indices = list(np.where(output_classes == cid)[0])
				data_slice = [nd_array_idx for elem_idx, nd_array_idx in enumerate(current_class_indices) if elem_idx < remaining_assigning_examples]
				current_class_input = input_data[data_slice]
				current_class_output = output_classes[data_slice]

				if np.size(partition_input) == 0 and np.size(partition_output) == 0:
					partition_input = current_class_input
					partition_output = current_class_output
				else:
					partition_input = np.concatenate((partition_input, current_class_input), axis=0)
					partition_output = np.concatenate((partition_output, current_class_output), axis=0)

				# Delete data after assignment from given input data
				input_data = np.delete(input_data, obj=data_slice, axis=0)
				output_classes = np.delete(output_classes, obj=data_slice, axis=0)
				current_size += np.size(current_class_output)

		# Increase counter for total assigned data
		total_examples_assigned += current_size
		partitions_datasets.append(PartitionDataset(input=partition_input,
													output=partition_output,
													partition_size=np.size(partition_output)))

	# Pass 2: Fill in all data left after pass 1
	# Sort partitions from largest size to smallest, so that we assign data based on skewness factor
	partitions_datasets = sorted(partitions_datasets, key=lambda pd: pd.partition_size, reverse=True)
	for cid in unique_classes:
		for pidx, partition_dataset in enumerate(partitions_datasets):
			current_class_indices = list(np.where(output_classes == cid)[0])
			remaining_class_examples = len(current_class_indices)
			if remaining_class_examples > 0:
				# Find number of class examples based on the skewness factor
				class_examples_for_partition = int(np.ceil(remaining_class_examples / skewness_factor))
				data_slice = current_class_indices[:class_examples_for_partition]
				# New data input for given partition
				add_partition_input = input_data[data_slice]
				add_partition_output = output_classes[data_slice]
				# Adjust new data input for the partition
				partitions_datasets[pidx].input = np.concatenate((partitions_datasets[pidx].input, add_partition_input), axis=0)
				partitions_datasets[pidx].output = np.concatenate((partitions_datasets[pidx].output, add_partition_output), axis=0)
				input_data = np.delete(input_data, obj=data_slice, axis=0)
				output_classes = np.delete(output_classes, obj=data_slice, axis=0)

	# Recompute data size and class distribution for each partition and create partition dataset mapping
	partitions_datasets_dict = dict()
	for pidx, partition_dataset in enumerate(partitions_datasets):
		partition_dataset.partition_size = np.size(partition_dataset.output)
		partition_dataset.compute_partition_distribution()
		partitions_datasets_dict[pidx] = partition_dataset

	# Sort partitions according to their data size and return a dictionary
	partitions_datasets = sorted(partitions_datasets, key=lambda pd: pd.partition_size)
	[print(x) for x in partitions_datasets]
	summation = sum([x.partition_size for x in partitions_datasets])
	print(summation)

	# Plot classes distribution
	classes = sorted(unique_classes)
	bar_x_values = [0] * num_partitions
	ind = np.arange(start=1, stop=num_partitions+1, step=1) # the x locations for the groups
	width = 0.35
	bars_pointers = []
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	for cid in classes:
		cid_bar_x_values = []
		partitions_datasets = sorted(partitions_datasets, key=lambda pd: pd.partition_size, reverse=True)
		for pd in partitions_datasets:
			partition_classes_stats = pd.partition_classes_stats
			not_found = True
			for p_cid in partition_classes_stats:
				if cid == p_cid:
					cid_size = partition_classes_stats[cid].class_number_of_examples
					cid_bar_x_values.append(cid_size)
					not_found = False
			if not_found:
				cid_bar_x_values.append(0)
		print(ind)
		print(cid_bar_x_values)
		print(width)
		print(bar_x_values)
		bar_pointer = ax.bar(ind, cid_bar_x_values, width, bottom=bar_x_values)
		bars_pointers.append(bar_pointer)
		bar_x_values = [sum(x) for x in zip(bar_x_values, cid_bar_x_values)]

	ax.set_xticks([x+1 for x in range(num_partitions)])
	ax.set_xlabel("Learner ID")
	ax.set_ylabel("Data per Learner")
	ax.set_title("Unbalanced Non-IID({}) Data Distribution, skewness: {}".format(classes_per_partition, skewness_factor))
	ax.legend(bars_pointers, classes, title='Classes')
	pdf_out = "unbalanced_data_distribution_partitions_{}_classes_per_partition_{}_skfactor_{}.pdf".format(num_partitions, classes_per_partition, skewness_factor)
	pdfpages1 = PdfPages(pdf_out)
	pdfpages1.savefig(figure=fig, bbox_inches='tight')
	pdfpages1.close()
	plt.show()

	return partitions_datasets_dict


if __name__=="__main__":
	# unbalanced_data_split_test()
	metis_db_session = MetisDBSession(mnist_session=True)
	metis_db_session.load_session_dataset(train_examples=3000, dev_examples=0, test_examples=0, distort_images=True)
	input_data = metis_db_session.get_input_trainset()
	output_data = metis_db_session.get_output_trainset()
	# unbalanced_data_partitioning(input=input_data, output=output_data, num_partitions=10, classes_per_partition=3, skewness_factor=1.3)
	metis_db_session.partition_session_training_data(partitions_num=10, classes_per_partition=3)
	res = metis_db_session.retrieve_partition_dataset_stats(to_json_representation=True)
	print(res)



