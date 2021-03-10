from collections import defaultdict
from collections import namedtuple
import numpy as np

PartitionDataset = namedtuple(typename='PartitionData', field_names=['distribution'])

class InformationEntropyOps(object):

	@staticmethod
	def compute_entropy_term(x, base=2):
		return -x * np.log(x) / np.log(base)

	@staticmethod
	def compute_etropy_per_partition(data_partitions):

		assert all(isinstance(value, PartitionDataset) for key, value in data_partitions.items())

		# collective data entropy, centralized data collection
		centralized_distribution = defaultdict(int)

		for pid, pdataset in data_partitions.items():
			for cid in pdataset.distribution:
				centralized_distribution[cid] += pdataset.distribution[cid]

		# Maximum Entropy
		max_entropy = np.log(len(centralized_distribution.keys())) / np.log(2)

		total_examples = np.sum([centralized_distribution[cid] for cid in centralized_distribution])
		centralized_percentages = dict()
		for cid in centralized_distribution:
			centralized_percentages[cid] = np.divide(centralized_distribution[cid], total_examples)

		# Divide by logging(2) to get entropy with base 2 logarithm
		centralized_entropy = np.sum([InformationEntropyOps.compute_entropy_term(centralized_percentages[cid], base=2) for cid in centralized_percentages])

		partitions_entropies = []
		average_entropy = 0
		intrinsic_information = 0
		for pid, pdataset in data_partitions.items():
			partition_examples = np.sum(list(pdataset.distribution.values()))
			partition_percentages = [np.divide(val, partition_examples) for val in pdataset.distribution.values()]
			partition_entropy = np.sum([InformationEntropyOps.compute_entropy_term(percentage, base=2) for percentage in partition_percentages])
			# Efficiency: Normalize by the maximum number of information
			# partition_entropy = np.sum([InformationEntropyOps.compute_entropy_term(percentage, base=2) / (np.logging(len(partition_percentages)) / np.logging(2)) for percentage in partition_percentages])
			partitions_entropies.append(partition_entropy)
			data_proportion = np.divide(partition_examples, total_examples)
			average_entropy += data_proportion * partition_entropy
			intrinsic_information += np.sum([InformationEntropyOps.compute_entropy_term(data_proportion)])

		information_gain = centralized_entropy - average_entropy
		gain_ratio = np.divide(information_gain, intrinsic_information)

		print("\nCentralized Entropy: {}\n\tPartitions Entropy: {}\n\tAverage Entropy: {}\n\tIntrinsic Information: {} \n\tInformation Gain: {} \n\tGain Ratio: {}"
			  .format(centralized_entropy, ["%.2f" % e for e in partitions_entropies], "%.2f" % average_entropy, "%.2f" % intrinsic_information, "%.2f" % information_gain, "%.2f" % gain_ratio))



if __name__=="__main__":

	partition_data = dict()

	# Total distribution:
	#  'A': 120
	#  'B': 60

	# Non-IID (1)
	partition_data[0] = PartitionDataset(distribution={'A': 120})
	partition_data[1] = PartitionDataset(distribution={'B': 60})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# Non-IID (1)-(2)
	partition_data[0] = PartitionDataset(distribution={'A': 60, 'B': 60})
	partition_data[1] = PartitionDataset(distribution={'A': 60})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# IID
	partition_data[0] = PartitionDataset(distribution={'A': 60, 'B': 30})
	partition_data[1] = PartitionDataset(distribution={'A': 60, 'B': 30})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# Total distribution:
	#  'A': 120
	#  'B': 60
	#  'C': 60

	# Non-IID (1)
	partition_data[0] = PartitionDataset(distribution={'A': 120})
	partition_data[1] = PartitionDataset(distribution={'B': 60})
	partition_data[2] = PartitionDataset(distribution={'C': 60})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# Non-IID (2)
	partition_data[0] = PartitionDataset(distribution={'A': 60, 'B': 20})
	partition_data[1] = PartitionDataset(distribution={'A': 60, 'B': 20})
	partition_data[2] = PartitionDataset(distribution={'B': 20, 'C': 60})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# Non-IID (1)-(2)
	partition_data[0] = PartitionDataset(distribution={'A': 60, 'B': 30})
	partition_data[1] = PartitionDataset(distribution={'A': 60, 'B': 30})
	partition_data[2] = PartitionDataset(distribution={'C': 60})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	# IID
	partition_data[0] = PartitionDataset(distribution={'A': 40, 'B': 20, 'C': 20})
	partition_data[1] = PartitionDataset(distribution={'A': 40, 'B': 20, 'C': 20})
	partition_data[2] = PartitionDataset(distribution={'A': 40, 'B': 20, 'C': 20})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)

	partition_data[0] = PartitionDataset(distribution={'A': 40, 'B': 30, 'C': 10})
	partition_data[1] = PartitionDataset(distribution={'A': 40, 'B': 30, 'C': 10})
	partition_data[2] = PartitionDataset(distribution={'A': 40, 'C': 40})
	InformationEntropyOps.compute_etropy_per_partition(partition_data)