from utils.logs_processing_plotting_ops import LogsProcessingPlotsUtil
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os


# BAR_PLOT_DEF = namedtuple('LearnersTrainingMetaBarPlot', field_names=['no_updates', 'no_epochs'])
STATS_DF_COLUMNS = ['Epochs', 'Updates']


def learners_stats_df(filepath):
	with open(filepath) as fin:
		json_federated_rounds = json.load(fin, object_pairs_hook=OrderedDict)

	# Define Learners Collections
	learners_total_completed_epochs = OrderedDict()
	learners_total_update_requests = OrderedDict()

	# Get learners ids as a well-defined and sorted collection
	learners_keys = LogsProcessingPlotsUtil.get_learners_keys(json_federated_rounds_obj=json_federated_rounds)
	learners_keys = LogsProcessingPlotsUtil.sort_alphanumeric(learners_keys, reverse=False)
	for learners_key in learners_keys:
		learners_total_completed_epochs[learners_key] = 0
	for learners_key in learners_keys:
		learners_total_update_requests[learners_key] = 0

	for fed_round_id in json_federated_rounds:
		fed_round_data = json_federated_rounds[fed_round_id]
		fdr_hosts = fed_round_data['hosts_results']

		for host_id in fdr_hosts:
			host_data = fdr_hosts[host_id]
			learner_key = LogsProcessingPlotsUtil.get_learner_plot_device_key(host_data['device_id'])
			learners_total_completed_epochs[learner_key] += float(host_data['completed_epochs'])
			learners_total_update_requests[learner_key] +=  int(host_data['host_to_controller_transmissions'])

	learners_training_stats_data = [ [learners_total_completed_epochs[learner_key], learners_total_update_requests[learner_key]] for learner_key in learners_keys]
	learners_training_stats_df = pd.DataFrame(data=learners_training_stats_data, columns=STATS_DF_COLUMNS)
	return learners_keys, learners_training_stats_df


def plot_epochs_and_updates_vbars(vbars_df, xtick_labels, fig_title='Clients trained epochs and updates requests'):

	assert isinstance(vbars_df, pd.DataFrame)
	fig = plt.figure()
	ax = fig.add_subplot(111)

	col1 = STATS_DF_COLUMNS[0]
	col2 = STATS_DF_COLUMNS[1]

	## the data
	N = len(vbars_df.index)
	epochs = vbars_df[col1]
	epochStd = epochs.std()
	updates = vbars_df[col2]
	updateStd = updates.std()

	## necessary variables
	ind = np.arange(N)                # the x locations for the groups
	width = 0.2                      # the width of the bars

	cmap = LogsProcessingPlotsUtil.get_colors_map(n=N, name='Set2')

	## bars definition
	rects1 = ax.bar(ind,
					epochs,
					width,
					color=cmap(0),
					edgecolor='black',
					hatch="\\\\\\")

	rects2 = ax.bar(ind+width,
					updates,
					width,
					color=cmap(2),
					edgecolor='black',
					hatch="--")

	# axes and labels
	ax.set_title(fig_title)
	ax.set_xlim(-width,len(ind)+width)
	# ax.set_ylim(0, epochs.)
	ax.set_xticks(ind+width)
	ax.set_xlabel('Clients')
	xtick_names = ax.set_xticklabels(xtick_labels)
	plt.setp(xtick_names, rotation=45, fontsize=10)

	## add a legend
	ax.legend( (rects1[0], rects2[0]), (col1, col2) )
	fig.subplots_adjust(bottom=0.3)

	return fig

if __name__=="__main__":

	scriptDirectory = os.path.dirname(os.path.realpath(__file__))
	cifar10_filepath = scriptDirectory + "/../resources/logs/testing_producibility/classes_10.clients_10F.Function_Di.SGDWithMomentum00.learningrate_01.batchsize_100.synchronous_False.targetexectimemins_25.BURNING_EPOCHS=0.UFREQUENCY=0.StratifiedHoldout5pct.VLossChange1pct.Adaptive10Requests.run_1.json"
	# mnist_filepath = scriptDirectory + "/../resources/logs/mnist_experiments/mnist.classes_10.fedrounds_10000.clients_3.batchsize_100.epochs_5.policyspecs.discountedfedavg_True.synchronous_True.targetlearners_6.targetepochs_None.targetaccuracy_None.targetexectimemins_30.json"

	filepath = cifar10_filepath

	learners_keys, learners_training_stats_df = learners_stats_df(filepath)

	print(learners_keys)
	print(learners_training_stats_df)

	fig_title = 'Clients trained epochs and update requests'
	fig = plot_epochs_and_updates_vbars(learners_training_stats_df, xtick_labels=learners_keys, fig_title=fig_title)
	pdf_name = "Mnist " + fig_title if 'mnist' in filepath else "Cifar10 " + fig_title
	# pdf_out = scriptDirectory + "/../resources/analysis_and_plotting/{}.pdf".format(pdf_name)
	# pdfpages = PdfPages(pdf_out)
	# pdfpages.savefig(figure=fig)
	# pdfpages.close()
	plt.show()