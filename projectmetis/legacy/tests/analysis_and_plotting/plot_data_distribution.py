from analysis_and_plotting.dvw.policies_convergence_rate_lines_plots import plot_federation_data_distribution
from matplotlib import pyplot as plt
import glob
import os

if __name__=="__main__":

	scriptDirectory = os.path.dirname(os.path.realpath(__file__))
	data_distributions_files = []
	# data_distributions_files.extend(glob.glob(os.path.join(scriptDirectory, "../../resources/logs/data_distributions/survey/extended_mnist/extended_mnist.classes_45.Skewness_0p0.DataDistribution.json")))
	# data_distributions_files.extend(glob.glob(os.path.join(scriptDirectory, "../../resources/logs/testing_producibility/*.json")))
	# data_distributions_files.extend(glob.glob(os.path.join(scriptDirectory, "../../resources/logs/testing_producibility/cifar100_data_distributions_survey/asynchronous_heterogeneous_cluster/cifar100.classes_50.BalancedRandom_False.BalancedClass_False.NISU_False.Skewness_1p3.clients_5F5S.Function_AsyncDVW*.json")))
	# data_distributions_files.extend(glob.glob(os.path.join(scriptDirectory, "../../resources/logs/testing_producibility/extended_mnist_by_class_data_distributions_survey/asynchronous_heterogeneous_cluster/emnist.classes_62.BalancedRandom_False.BalancedClass_False.NISU_False.Skewness_1p5.clients_5F5S.Function_AsyncDVW*.json")))
	# data_distributions_files.extend(glob.glob("../metisdb/cifar100.classes_50.Skewness_1p3.Skewed.DataDistribution.json"))
	# data_distributions_files.extend(glob.glob("../metisdb/extended_mnist.classes_30.Skewness_0p0.Skewed.DataDistribution.json"))
	data_distributions_files.extend(glob.glob("../metisdb/extended_mnist.classes_62.Skewness_1p5.Skewed.DataDistribution.json"))


	for data_distribution_file in data_distributions_files:
		fig_name=None
		fig, lgd = plot_federation_data_distribution(data_distribution_file, fig_name=fig_name)
		output_filename = "{}_{}_{}_DataDistribution".format(
			"Cifar10", "Uniform", "IID"
		)
		file_out = os.path.join(scriptDirectory, "../../resources/plots/{}.png".format(output_filename))

		if lgd is not None:
			plt.savefig(file_out, bbox_extra_artists=(lgd,), bbox_inches='tight')
		else:
			plt.savefig(file_out, bbox_inches='tight')

	plt.show()