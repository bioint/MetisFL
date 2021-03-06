import random

import numpy as np

from collections import namedtuple, Counter, defaultdict
from deprecated import deprecated
from utils.logging.metis_logger import MetisLogger as metis_logger

ShardDataset = namedtuple(typename='ShardData', field_names=['input', 'output'])

class PartitionDatasetClassStats(object):
	def __init__(self, class_id, class_number_of_examples, class_examples_percentage):
		self.class_id = int(class_id)
		self.class_number_of_examples = int(class_number_of_examples)
		self.class_examples_percentage = float(class_examples_percentage)

	def __repr__(self):
		return "<class id: {}, number of examples: {}, examples percentage: {}>".format(self.class_id,
																						self.class_number_of_examples,
																						self.class_examples_percentage)

	def toJSON_representation(self):
		return { "class_id": self.class_id,
				 "class_number_of_examples": self.class_number_of_examples,
				 "class_examples_percentage": self.class_examples_percentage }

class PartitionDataset(object):

	def __init__(self, input, output, partition_size, partition_classes_stats=None):
		self.input = input
		self.output = output
		self.partition_size = int(partition_size)
		self.partition_classes = [int(cid) for cid in np.unique(self.output)]
		self.partition_classes_stats = partition_classes_stats
		self.training_data_size = self.partition_size
		if partition_classes_stats is None:
			self.compute_partition_distribution()

	def compute_partition_distribution(self):
		data_counter = Counter(self.output)
		partition_class_distribution = dict()
		for key in data_counter.keys():
			partition_class_distribution[key] = PartitionDatasetClassStats(class_id=key,
																		   class_number_of_examples=data_counter[key],
																		   class_examples_percentage=float(round(data_counter[key] / self.partition_size, 2)))
		# Sort partition distribution in ascending order of the class_id
		sorted(partition_class_distribution)
		self.partition_classes_stats = partition_class_distribution


	def __repr__(self):
		data_distribution_repr = [self.partition_classes_stats[cid] for cid in sorted(self.partition_classes_stats.keys())]
		return "Input Data Shape: {}, " \
			   "Output Data Shape: {}, " \
			   "Partition Size: {}, " \
			   "Partition Classes: {}, " \
			   "Partition Classes Number: {}, " \
			   "Partition Data Distribution: {}".format(self.input.shape,
														self.output.shape,
														self.partition_size,
														self.partition_classes,
														len(self.partition_classes),
														data_distribution_repr)

	def toJSON_representation(self):
		jsonifed_partition_classes_stats = [self.partition_classes_stats[cid].toJSON_representation() for cid in sorted(self.partition_classes_stats.keys())]
		return { "partition_size": self.partition_size,
				 "partition_classes": self.partition_classes,
				 "partition_classes_stats": jsonifed_partition_classes_stats}


class DataPartitioningUtil(object):

	@staticmethod
	def balanced_random_partitioning(input_data, output_classes, partitions=100, ignore_output_class_from_stats=None):
		"""
		Partitions the input data in a iid fashion.
		Randomly shuffle the data and then assign them to partitions.
		Returns a dictionary with the partition_id as the key and PartitionDataset namedtuple as the value
		:param input_data:
		:param output_classes:
		:param partitions:
		:return:
		"""
		if not ((isinstance(input_data, list) and isinstance(output_classes, list)) or
				(isinstance(input_data, np.ndarray) and isinstance(output_classes, np.ndarray))):
			raise TypeError("`input_data` and `output_classes` arguments must both be of type %s or %s "
							"but instead they were %s and %s" % (list, type(np.ndarray), type(input_data), type(output_classes)))
		if len(input_data) != len(output_classes):
			raise ValueError("`input_data` and `output_classes` must have the same size")

		total_data = len(input_data)
		data_indices = list(range(total_data))

		# shuffle indexes instead of actual list
		np.random.shuffle(data_indices)

		# how many data to assign per client
		# use ceil to avoid not placing data to the clients
		data_per_client = int(np.ceil(total_data / partitions))

		partitions_datasets = dict()
		for pid in range(partitions):
			low = pid * data_per_client
			high = low + data_per_client

			partition_data_indices = data_indices[slice(low, high)]
			partition_data_size = len(partition_data_indices)

			# handle partition input and output as lists
			partition_input = [input_data[idx] for idx in partition_data_indices] if isinstance(input_data, list) else input_data[partition_data_indices]
			partition_output = [output_classes[idx] for idx in partition_data_indices] if isinstance(output_classes, list) else output_classes[partition_data_indices]

			# Case where the output classes is a list of nested lists. e.g.
			# output_classes: [
			# 	[8,7,6,5,4,7] -> record1
			#	[1,2,3,1,5,7,8] -> record2
			# ]
			if any(isinstance(classes, list) for classes in partition_output):
				percentages = Counter()
				for record_classes in partition_output:
					record_classes = [record_class for record_class in record_classes if record_class != ignore_output_class_from_stats]
					record_percentages = Counter(record_classes)
					percentages += record_percentages
			else:
				percentages = Counter(partition_output)

			for key in percentages.keys():
				percentages[key] = PartitionDatasetClassStats(class_id=key,
															  class_number_of_examples=percentages[key],
															  class_examples_percentage=round(percentages[key] / partition_data_size, 2))

			partitions_datasets[pid] = PartitionDataset(input=partition_input,
														output=partition_output,
														partition_size=np.size(partition_output),
														partition_classes_stats=percentages)

		return partitions_datasets


	@staticmethod
	@deprecated(reason='This is equivalent to noniid_partitioning function, when the number of classes per partition is equal to 2')
	def extreme_noniid_partitioning(input_data, output_classes, sort_by_class=True, order=None, partitions=100, shards_num=200):
		"""
		This function sorts the input data by label, creates the respective number of shards and then assigns sequentially the shards to each partition
		:param input_data:
		:param output_classes:
		:param sort_by_class:
		:param order:
		:param partitions:
		:param shards_num:
		:return:
		"""

		if np.size(input_data, 0) != output_classes.size:
			raise ValueError("`input_data` x-axis size (the number of examples) and `output_classes` must have the same size")
		if not (isinstance(input_data, np.ndarray) and isinstance(output_classes, np.ndarray)):
			raise TypeError("`input_data` and `output_classes` arguments must both be of type %s or %s "
							"but instead they were %s and %s" % (list, type(np.ndarray), type(input_data), type(output_classes)))
		if not sort_by_class and order is None:
			raise ValueError("When `sort_by_class` is False, a value for the `order` must be given")

		if sort_by_class:
			permutation = output_classes.argsort(kind="mergesort")
			input_data = input_data[permutation]
			output_classes = output_classes[permutation]
		else:
			pass
		# TODO Need to add support for sorting based on values of ndarray input data (i.e. combinations of (input_data & order))

		total_examples = output_classes.size
		data_indices = list(range(total_examples))
		shard_size = int(total_examples / shards_num)
		shards_per_client = int(total_examples / (shard_size * partitions))
		indices_buckets = list()

		# create a bucket with indices per shard, each index inside the buckets refers to a data point in the original data collection
		for shardid in range(0, shards_num):
			low = shardid * shard_size
			high = low + shard_size
			indices_buckets.append([data_indices[idx] for idx in range(low, high)])

		partitions_datasets = dict()
		# Assign to each partition its respective number of shards
		for pid in range(partitions):
			low = pid * shards_per_client
			high = low + shards_per_client

			partition_data_indices = [dindex for list_index in indices_buckets[slice(low, high)] for dindex in list_index]
			partition_data_size = len(partition_data_indices)

			partition_input = input_data[partition_data_indices]
			partition_output = output_classes[partition_data_indices]
			percentages = Counter(partition_output)
			for key in percentages.keys():
				percentages[key] = PartitionDatasetClassStats(class_id=key,
															  class_number_of_examples=percentages[key],
															  class_examples_percentage=round(percentages[key] / partition_data_size, 2))

			partitions_datasets[pid] = PartitionDataset(input=partition_input,
														output=partition_output,
														partition_size=partition_data_size,
														partition_classes_stats=percentages)

		for pid in range(partitions):
			sorted_by_key_percentages = sorted(partitions_datasets[pid].partition_examples_stats.items())
			partition_data_size = partitions_datasets[pid].partition_size
			metis_logger.info('Partition Number: %s, Partition Data Size: %s, Training Examples Stats: %s' % (pid, partition_data_size, sorted_by_key_percentages))

		return partitions_datasets


	@staticmethod
	@deprecated(reason='This is handled now by unbalanced or balanced and number of classes per partition')
	def shard_based_noniid_partitioning(input_data, output_classes, partitions=10, classes_per_partition=3, shards_num=200):

		if np.size(input_data, 0) != output_classes.size:
			raise ValueError("`input_data` x-axis size (the number of examples) and `output_classes` must have the same size")
		if not (isinstance(input_data, np.ndarray) and isinstance(output_classes, np.ndarray)):
			raise TypeError("`input_data` and `output_classes` arguments must both be of type %s or %s "
							"but instead they were %s and %s" % (list, type(np.ndarray), type(input_data), type(output_classes)))

		# TODO Need to add support for sorting based on values of ndarray input data (i.e. combinations of (input_data & order))
		# At first, we sort the data based on their classes and then we proceed
		permutation = output_classes.argsort(kind="mergesort")
		input_data = input_data[permutation]
		output_classes = output_classes[permutation]

		examples_per_class = Counter(output_classes)
		total_examples = output_classes.size
		data_indices = list(range(total_examples))
		shard_size = int(total_examples / shards_num)
		total_shards_per_client = int(total_examples / (shard_size * partitions))

		# create a bucket with indices per shard, each index inside the buckets refers to a data point in the original data collection
		shards = {}
		class_shards = defaultdict(lambda: list(), {})
		for shardid in range(0, shards_num):
			low = shardid * shard_size
			high = low + shard_size
			shard_data_indices = [data_indices[idx] for idx in range(low, high)]

			shard_dataset = ShardDataset(input=input_data[shard_data_indices],
										 output=output_classes[shard_data_indices])
			shards[shardid] = shard_dataset
			shard_majority_class = Counter(shard_dataset.output).most_common()[0][0]
			class_shards[shard_majority_class].append(shardid)

		total_class_shards_per_partition = int(total_shards_per_client / classes_per_partition)
		partitions_datasets = dict()
		# Assign to each partition its respective number of shards
		for pid in range(partitions):
			partition_shards_ids = list()
			num_partition_remaining_shards = total_shards_per_client

			while num_partition_remaining_shards > 0:
				for cid in class_shards.keys():
					current_class_partition_shards = class_shards[cid][:total_class_shards_per_partition]
					# Update Partition's Shards counters
					current_selection_size = len(current_class_partition_shards)
					num_partition_remaining_shards -= current_selection_size

					if num_partition_remaining_shards < 0:
						current_class_partition_shards = current_class_partition_shards[:num_partition_remaining_shards]
						num_partition_remaining_shards = 0

					class_shards[cid] = [shardid for shardid in class_shards[cid] if shardid not in current_class_partition_shards]
					partition_shards_ids.extend(current_class_partition_shards)
					if num_partition_remaining_shards == 0:
						break

			partition_input = shards[partition_shards_ids[0]].input
			partition_output = shards[partition_shards_ids[0]].output
			for sid in partition_shards_ids[1:]:
				shard_input = shards[sid].input
				shard_output = shards[sid].output
				partition_input = np.concatenate((partition_input, shard_input), axis=0)
				partition_output = np.concatenate((partition_output, shard_output), axis=0)

			partition_data_size = partition_output.size
			percentages = Counter(partition_output)
			for key in percentages.keys():
				percentages[key] = PartitionDatasetClassStats(class_id=key,
															  class_number_of_examples=percentages[key],
															  class_examples_percentage=round(percentages[key] / partition_data_size, 2))

			partitions_datasets[pid] = PartitionDataset(input=partition_input,
														output=partition_output,
														partition_size=partition_data_size,
														partition_classes_stats=percentages)

		for pid in range(partitions):
			partition_data_size = partitions_datasets[pid].partition_size
			sorted_by_key_percentages = sorted(partitions_datasets[pid].partition_examples_stats.items())
			metis_logger.info('Partition Number: %s, Partition Data Size: %s, Training Examples Stats: %s' % (pid, partition_data_size, sorted_by_key_percentages))

		return partitions_datasets


	@staticmethod
	def balanced_class_partitioning(input_data, output_classes, partitions=10, classes_per_partition=5):

		# At first, we sort the data based on their classes and then we proceed
		permutation = output_classes.argsort(kind="mergesort")
		input_data = input_data[permutation]
		output_classes = output_classes[permutation]
		examples_per_class = Counter(output_classes)

		# Assign Classes to each Partition RoundRobin
		classes_set = list(set(output_classes))
		total_classes = len(classes_set)
		# Get total number of examples.
		num_examples = np.size(output_classes)

		partition_class_assignments = {}
		class_partition_assignments = defaultdict(int)

		# Different offset for different domains.
		# E.g. offset equal to 1 is Round-Robin assignment of classes per client
		# Cifar10, offset=1
		# Cifar100, offset=25
		# ExtednedMnistByClass, offset=25
		if len(classes_set) == 10:
			# offset = 1
			# RoundRobin proportionally to the number of classes and partitions
			offset = int(len(classes_set)/partitions)
		elif len(classes_set) == 100:
			# offset = 10
			# RoundRobin proportionally to the number of classes and partitions
			offset = int(100/partitions)
		elif len(classes_set) == 62:
			# offset = classes_per_partition
			# offset = 25
			offset = 6
		else:
			offset = 1

		for pid in range(partitions):
			# Get number of classes each partition must hold
			partition_assignment = classes_set[:classes_per_partition]
			# Find to how many partitions each class must be assigned to
			for c in partition_assignment:
				class_partition_assignments[c] += 1
				# Get the first element from the list, delete it and append it at the end
			for i in range(offset):
				elem = classes_set[0]
				del classes_set[0]
				classes_set.append(elem)
			partition_class_assignments[pid] = partition_assignment

		partition_data_indices = defaultdict(lambda: list(), {})
		for cid, c_assignments in class_partition_assignments.items():
			current_class_indices = np.where(output_classes == cid)[0]
			current_class_datasize = current_class_indices.size
			# Find how many data from each class must be assigned to every partition
			class_data_per_partition = int(current_class_datasize / c_assignments)

			class_start_index = current_class_indices[0]
			class_end_index = class_start_index + class_data_per_partition - 1
			so_far_assigned = 0
			for pid, p_assignments in partition_class_assignments.items():
				if cid in p_assignments:
					partition_data_indices[pid].append((class_start_index, class_end_index))
					so_far_assigned = so_far_assigned + class_end_index - class_start_index + 1
					class_start_index = class_end_index + 1
					class_end_index = class_end_index + class_data_per_partition

			if so_far_assigned < current_class_datasize:
				remaining = current_class_datasize - so_far_assigned
				last_indices = current_class_indices[-remaining:]
				# Shuffle the partition ids so that we assign the remaining data randomly to one partition
				pids = list(partition_class_assignments.keys())
				random.shuffle(pids)
				for pid in pids:
					p_assignments = partition_class_assignments[pid]
					if cid in p_assignments:
						partition_data_indices[pid].append((last_indices[0], last_indices[-1]))
						break

		# Create the input and output datasets of the partitions
		total_examples_assigned = 0
		partitions_datasets = dict()
		for pid in partition_class_assignments.keys():
			partition_indices = partition_data_indices[pid]
			low = partition_indices[0][0]
			high = partition_indices[0][1]
			data_slice = [idx for idx in range(low, high + 1)]
			partition_input = input_data[data_slice]
			partition_output = output_classes[data_slice]
			for indices in partition_indices[1:]:
				low = indices[0]
				high = indices[1]
				data_slice = [idx for idx in range(low, high + 1)]
				partition_input = np.concatenate((partition_input, input_data[data_slice]), axis=0)
				partition_output = np.concatenate((partition_output, output_classes[data_slice]), axis=0)

			partition_data_size = partition_output.size
			total_examples_assigned += partition_data_size

			# Partition input and output must be shuffled because currently they are all sorted by class
			indices = np.arange(partition_input.shape[0])
			np.random.shuffle(indices)
			partition_input = partition_input[indices]
			partition_output = partition_output[indices]
			partitions_datasets[pid] = PartitionDataset(input=partition_input,
														output=partition_output,
														partition_size=partition_data_size)

		if total_examples_assigned != num_examples:
			raise RuntimeError("The number of examples assigned to all partitions does not match "
							   "the original number of training examples Original vs Assigned: {} vs {}".format(
				num_examples, total_examples_assigned
			))

		return partitions_datasets


	@staticmethod
	def strictly_unbalanced_noniid_data_partitioning(input_data, output_classes, partitions, classes_per_partition, skewness_factor):

		if skewness_factor == 0:
			raise RuntimeError("Skewness value cannot be equal to 0")

		assert isinstance(input_data, np.ndarray)
		assert isinstance(output_classes, np.ndarray)

		# Sort data according to class id.
		permutation = output_classes.argsort(kind="mergesort")
		input_data = input_data[permutation]
		output_classes = output_classes[permutation]

		# Get total number of examples.
		num_examples = np.size(output_classes)
		unique_classes = list(np.unique(output_classes))

		# Find smallest bin size based on the factor difference between two consecutive partitions.
		# CAUTION pidx starts from 0, thus +1 (need it for factorized bin sizes)
		factorized_bin_sizes = [np.power(skewness_factor, pidx+1) for pidx in range(partitions)]
		bin_size_factor = np.floor(num_examples / sum(factorized_bin_sizes))

		# Find actual bin sizes using the smallest bin size (bin_size_factor).
		partitions_sizes = [np.floor(np.power(skewness_factor, pidx+1) * bin_size_factor) for pidx in range(partitions)]
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
					# Find the slice
					data_slice = [nd_array_idx for elem_idx, nd_array_idx in enumerate(current_class_indices) if elem_idx < remaining_assigning_examples]
					current_class_input = input_data[data_slice]
					current_class_output = output_classes[data_slice]

					if np.size(partition_input) == 0 and np.size(partition_output) == 0:
						partition_input = current_class_input
						partition_output = current_class_output
					else:
						partition_input = np.concatenate((partition_input, current_class_input), axis=0)
						partition_output = np.concatenate((partition_output, current_class_output), axis=0)

					# Delete data after assignment from given input_data data
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
		total_examples_assigned = 0
		partitions_datasets_dict = dict()
		for pidx, partition_dataset in enumerate(partitions_datasets):
			partitions_datasets_dict[pidx] = PartitionDataset(input=partition_dataset.input,
															  output=partition_dataset.output,
															  partition_size=np.size(partition_dataset.output))
			total_examples_assigned += partitions_datasets_dict[pidx].partition_size

		if total_examples_assigned != num_examples:
			raise RuntimeError("The number of examples assigned to all partitions does not match "
							   "the original number of training examples Original vs Assigned: {} vs {}".format(
				num_examples, total_examples_assigned
			))

		return partitions_datasets_dict


	@staticmethod
	def strictly_noniid_unbalanced_data_partitioning(input_data, output_classes, partitions, classes_per_partition, skewness_factor):

		if skewness_factor == 0:
			raise RuntimeError("Skewness value cannot be equal to 0")

		assert isinstance(input_data, np.ndarray)
		assert isinstance(output_classes, np.ndarray)

		# Sort data according to class id.
		permutation = output_classes.argsort(kind="mergesort")
		input_data = input_data[permutation]
		output_classes = output_classes[permutation]

		# Get total number of examples.
		num_examples = np.size(output_classes)
		unique_classes = list(np.unique(output_classes))
		total_classes_num = len(unique_classes)

		if len(unique_classes) < classes_per_partition:
			raise RuntimeError("Number of available classes is less than the number of partitions")

		# Find smallest bin size based on the factor difference between two consecutive partitions.
		# Caution idx starts from 0, thus +1
		factorized_bin_sizes = [np.power(skewness_factor, idx+1) for idx in range(partitions)]
		bin_size_factor = np.ceil(num_examples / sum(factorized_bin_sizes))

		# Find actual bin sizes using the smallest bin size.
		partitions_sizes = [np.ceil(np.power(skewness_factor, idx+1) * bin_size_factor) for idx in range(partitions)]
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
			# Bring the partitions into descending size order (largest->smallest)
			partitions_sizes = list(reversed(new_partitions_sizes))


		# Sentinel dictionary that takes into account the true number of examples per class
		#  that allows even distribution of class examples across partitions. With this approach
		#  we can make sure that for the requested number of classes per partition,
		#  each partition will hold data from every such class
		allowed_examples_per_class_per_partition = dict()
		for cid in unique_classes:
			allowed_examples = np.floor(np.divide(
				np.where(output_classes == cid)[0].size,
				partitions))
			allowed_examples_per_class_per_partition[cid] = allowed_examples


		partitions_datasets = []
		already_assigned_classes = set()
		# Pass 1: Fill in the partition buckets
		# 	a. Loop through the classes and assign examples to partitions
		#	b. Constraint: No partition has more classes than the given `classes_per_partition`
		# Assignment tales place from the smallest sized partition to the largest
		# The following examples filling considers the partition size and the number of classes per partition
		#  to initialize data filling.
		for pid, psize in enumerate(partitions_sizes):

			current_size = 0
			partition_input = list()
			partition_output = list()
			current_partition_classes_num = 0

			# Number of examples per class based on partition size and number of classes per partition
			examples_per_class = int(np.floor(psize / classes_per_partition))

			# Shuffle classes id to add randomness in our partition distribution
			random.shuffle(unique_classes)

			while current_partition_classes_num < classes_per_partition:

				cid = unique_classes.pop(0)

				# Ensure that all classes get assigned
				# (some classes may not be assigned due to random shuffling)
				if len(already_assigned_classes) < total_classes_num:
					while cid in already_assigned_classes:
						unique_classes.append(cid)
						cid = unique_classes.pop(0)

				allowed_examples = min(examples_per_class, allowed_examples_per_class_per_partition[cid])
				partition_remaining_slots = psize - current_size
				remaining_assigning_examples = int(min(partition_remaining_slots, allowed_examples))
				current_class_indices = list(np.where(output_classes == cid)[0])
				# Find the slice
				data_slice = [nd_array_idx for elem_idx, nd_array_idx in enumerate(current_class_indices)
							  if elem_idx < remaining_assigning_examples]
				current_class_input = input_data[data_slice]
				current_class_output = output_classes[data_slice]

				if np.size(partition_input) == 0 and np.size(partition_output) == 0:
					partition_input = current_class_input
					partition_output = current_class_output
				else:
					partition_input = np.concatenate((partition_input, current_class_input), axis=0)
					partition_output = np.concatenate((partition_output, current_class_output), axis=0)

				# Delete data after assignment from given input_data data
				input_data = np.delete(input_data, obj=data_slice, axis=0)
				output_classes = np.delete(output_classes, obj=data_slice, axis=0)
				current_size += np.size(current_class_output)
				current_partition_classes_num = np.unique(partition_output).size
				unique_classes.append(cid)
				already_assigned_classes.add(cid)

			partitions_datasets.append(PartitionDataset(input=partition_input,
														output=partition_output,
														partition_size=np.size(partition_output)))

		# Pass 2: Fill in all data left after pass 1
		# For each remaining class find the partitions owning the class
		# Assign the remaining number of examples of each class
		#  to each partition based on skewness
		remaining_classes = list(np.unique(output_classes))
		remaining_classes_assignment = defaultdict(list)
		for cid in remaining_classes:
			for pidx, partition_dataset in enumerate(partitions_datasets):
				if cid in list(np.unique(partitions_datasets[pidx].output)):
					remaining_classes_assignment[cid].append(pidx)

		for cid, cid_assignee_partitions in remaining_classes_assignment.items():
			remaining_class_number_of_examples = np.where(output_classes == cid)[0].size

			# CAUTION pidx starts from 0, thus +1 (need it for factorized bin sizes)
			factorized_bin_sizes = [np.ceil(np.power(skewness_factor, pidx+1)) for pidx in cid_assignee_partitions]
			bin_size_factor = np.ceil(remaining_class_number_of_examples / sum(factorized_bin_sizes))
			class_assignee_partitions_examples_num = [(pidx, int(np.ceil(np.ceil(np.power(skewness_factor, pidx+1)) * bin_size_factor)))
													  for pidx in cid_assignee_partitions]

			for pidx, assignee_partition_examples_num in class_assignee_partitions_examples_num:
				remaining_class_examples_indices = list(np.where(output_classes == cid)[0])
				data_slice = remaining_class_examples_indices[:assignee_partition_examples_num]
				add_partition_input = input_data[data_slice]
				add_partition_output = output_classes[data_slice]
				partitions_datasets[pidx].input = np.concatenate((partitions_datasets[pidx].input,
																  add_partition_input), axis=0)
				partitions_datasets[pidx].output = np.concatenate((partitions_datasets[pidx].output,
																   add_partition_output), axis=0)
				input_data = np.delete(input_data, obj=data_slice, axis=0)
				output_classes = np.delete(output_classes, obj=data_slice, axis=0)


		partitions_datasets = sorted(partitions_datasets, key=lambda pd: pd.partition_size, reverse=True)
		# Recompute data size and class distribution for each partition and create partition dataset mapping
		total_examples_assigned = 0
		partitions_datasets_dict = dict()
		for pidx, partition_dataset in enumerate(partitions_datasets):
			partitions_datasets_dict[pidx] = PartitionDataset(input=partition_dataset.input,
															  output=partition_dataset.output,
															  partition_size=np.size(partition_dataset.output))
			total_examples_assigned += partitions_datasets_dict[pidx].partition_size

		if total_examples_assigned != num_examples:
			raise RuntimeError("The number of examples assigned to all partitions does not match "
							   "the original number of training examples Original vs Assigned: {} vs {}".format(
				num_examples, total_examples_assigned
			))

		# It is required to return a dictionary thus the conversion from list to dict above
		return partitions_datasets_dict


	@staticmethod
	def stratified_training_holdout_dataset(partition_data, holdout_proportion):
		partition_input_data = partition_data.input
		partition_output_data = partition_data.output

		# Make sure the partition data have the correct format(i.e, np_arrays), in order to be manipulated
		if isinstance(partition_input_data, list):
			partition_input_data = np.array(partition_input_data)
		if isinstance(partition_output_data, list):
			partition_output_data = np.array(partition_output_data)

		partition_size = partition_data.partition_size
		original_training_size = partition_size
		partition_classes_stats = partition_data.partition_classes_stats

		# The classes/labels of the training examples that this partition holds
		partition_class_ids = partition_classes_stats.keys()
		# Number of unique classes in this partition
		partition_number_of_classes = len(partition_class_ids)

		# Create holdout dataset
		holdout_set_idxs = np.array([], dtype=int)
		for class_id in partition_class_ids:
			class_id_examples_idx = np.where(partition_output_data == class_id)[0]
			# Number/Proportion of examples that each class/label must contribute to the holdout dataset
			class_id_holdout_examples = np.floor(class_id_examples_idx.size * holdout_proportion).astype(int)
			# Pick the elements of the holdout set randomly
			# holdout_class_id_idxs = np.random.choice(class_id_examples_idx, class_id_holdout_examples, replace=False)
			# Pick the elements of the holdout set sequentially
			holdout_class_id_idxs = np.take(class_id_examples_idx, range(class_id_holdout_examples))
			holdout_set_idxs = np.append(holdout_set_idxs, holdout_class_id_idxs)

		# shuffle once to avoid indexes of the same class being one next to the other in the holdout set
		# np.random.shuffle(holdout_set_idxs)

		# Finalize training and holdout datasets
		holdout_set_input = partition_input_data[holdout_set_idxs]
		holdout_set_output = partition_output_data[holdout_set_idxs]
		holdout_set_size = holdout_set_output.size

		training_set_input = np.delete(partition_input_data, holdout_set_idxs, axis=0)
		training_set_output = np.delete(partition_output_data, holdout_set_idxs, axis=0)
		training_set_size = training_set_output.size

		training_partition_dataset = PartitionDataset(input=training_set_input,
													  output=training_set_output,
													  partition_size=training_set_size)
		validation_partition_dataset = PartitionDataset(input=holdout_set_input,
														output=holdout_set_output,
														partition_size=holdout_set_size)

		return training_partition_dataset, validation_partition_dataset

