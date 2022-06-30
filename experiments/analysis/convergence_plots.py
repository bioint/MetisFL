import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


def convert_to_timestamp(t):
    return datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ').timestamp()


def plot_brainage_convergence(files, metric="mae", show_global_models_exchanged=False, show_processing_time=False):

    BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS = [
        "uniform_iid", "uniform_noniid",
        "skewed135_iid", "skewed135_noniid"
    ]
    BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS_SUBPLOTS_TITLES = [
        "Uniform & IID", "Uniform & Non-IID",
        "Skewed & IID", "Skewed & Non-IID"
    ]
    BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS_AXIS_PLACEHOLDERS = [
        (0, 0), (0, 1),
        (1, 0), (1, 1)
    ]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
    fig.subplots_adjust(bottom=0.2, wspace=0.1, hspace=0.3)

    plt.rcParams['axes.titlesize'] = 32

    for environment_idx, (dist_id, subplot_ttl, axis_ph) in \
        enumerate(zip(BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS,
                      BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS_SUBPLOTS_TITLES,
                      BRAINAGE_DATA_DISTRIBUTIONS_IDENTIFIERS_AXIS_PLACEHOLDERS)):
        print(environment_idx, dist_id, subplot_ttl, axis_ph)
        ax[axis_ph].set_title(subplot_ttl)
        ax[axis_ph].grid(True)
        ax[axis_ph].set_ylim(2.5, 6.5)
        ax[axis_ph].tick_params(labelsize=32)

        x_axis, y_axis = [], []
        for file in files:
            if dist_id in file:
                print(dist_id, file, flush=True)
                json_data = json.load(open(file, "r"))
                federation_runtime_metadata = json_data["federation_runtime_metadata"]["metadata"]
                completed_global_iterations = \
                    [(iteration["global_iteration"], convert_to_timestamp(iteration["completed_at"]))
                     for iteration in federation_runtime_metadata if "completed_at" in iteration]
                global_iterations_sorted = sorted(completed_global_iterations, key=lambda x: x[1])  # sort by timestamp
                initial_timestamp = global_iterations_sorted[0][1]
                # print([x[1]-initial_timestamp for x in global_iterations_sorted[1:]], flush=True)
                # default is seconds - /60 to minutes
                global_iterations = [(x[0], x[1] - initial_timestamp) for x in global_iterations_sorted]
                global_iterations_index = [x[0] for x in global_iterations]
                global_iterations_ts = [x[1] for x in global_iterations]

                community_evaluations = json_data["community_model_results"]["community_evaluation"]

                global_model_test_performances = []
                for iter_id, ts in global_iterations:
                    iteration_performances = []
                    for evaluation in community_evaluations:
                        global_iteration_id = evaluation["global_iteration"]
                        if iter_id == global_iteration_id and "evaluations" in evaluation:
                            evaluations = evaluation["evaluations"]
                            for learner_id, learner_evaluation in evaluations.items():
                                iteration_performances.append(
                                    float(learner_evaluation["test_evaluation"]["metric_values"][metric]))
                    if len(iteration_performances) > 0:
                        iteration_performance = np.mean(iteration_performances)
                        global_model_test_performances.append(iteration_performance)
                global_iterations_index = global_iterations_index[:len(global_model_test_performances)]
                global_iterations_ts = [x for x in global_iterations_ts if x <= 20000]

                if show_global_models_exchanged:
                    x_axis, y_axis = global_iterations_index, global_model_test_performances[
                                                              :len(global_iterations_index)]
                elif show_processing_time:
                    x_axis, y_axis = global_iterations_ts, global_model_test_performances[:len(global_iterations_ts)]
                line_label = None
                linestyle = "solid"
                if "_AsyncFedAvg_" in file:
                    line_label = "AsyncFedAvg"
                    color = "orange"
                elif "_SyncFedAvg_" in file:
                    line_label = "Sync"
                    color = "darkblue"
                elif "_SemiSyncFedAvg_" in file:
                    lambda_val = int(file.split("_lambda")[1].split("_")[0])
                    if lambda_val == 2:
                        color = "dodgerblue"
                    elif lambda_val == 4:
                        color = "forestgreen"
                    line_label = "SemiSync ($\lambda={}$)".format(lambda_val)
                    linestyle = "dashed"

                if environment_idx != 0:
                    line_label = None
                ax[axis_ph].plot(x_axis, y_axis, linestyle=linestyle, linewidth=4, label=line_label, color=color)

        # 3d model centralized: 2.694456259
        # 2d model centralized: 2.667021116
        line_label = None
        if environment_idx == 0:
            line_label = "Centralized"
        ax[axis_ph].plot(x_axis, len(x_axis) * [2.694456259],
                         linestyle="solid", color='crimson',
                         linewidth=4, label=line_label)

    fig.text(0.09, 0.5, 'MAE', ha='center', va='center', rotation='vertical', fontsize=32)
    if show_processing_time:
        fig.text(0.5, 0.14, 'Processing Time(secs)', ha='center', va='center', fontsize=32)
    if show_global_models_exchanged:
        fig.text(0.5, 0.14, '# Global Models Exchanged', ha='center', va='center', fontsize=32)

    fig.legend(loc='lower center', fancybox=False, shadow=False, fontsize=32, ncol=3)
    return fig


def plot_rounds_convergence(files, metric="accuracy"):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for file in files:

        line_label = "global model"
        print(file, flush=True)
        json_data = json.load(open(file, "r"))
        federation_runtime_metadata = json_data["federation_runtime_metadata"]["metadata"]
        completed_global_iterations = \
            [(iteration["global_iteration"], convert_to_timestamp(iteration["completed_at"]))
             for iteration in federation_runtime_metadata if "completed_at" in iteration]
        global_iterations_sorted = sorted(completed_global_iterations, key=lambda x: x[1])  # sort by timestamp
        initial_timestamp = global_iterations_sorted[0][1]
        print([x[1] - initial_timestamp for x in global_iterations_sorted[1:]], flush=True)
        # default is seconds - /60 to minutes
        global_iterations = [(x[0], x[1] - initial_timestamp) for x in global_iterations_sorted]
        global_iterations_index = [x[0] for x in global_iterations]
        global_iterations_ts = [x[1] for x in global_iterations]

        community_evaluations = json_data["community_model_results"]["community_evaluation"]
        learners_task_results = json_data["learners_models_results"]["learner_task"]

        global_model_test_performances = []
        for iter_id, ts in global_iterations:
            iteration_performances = []
            for evaluation in community_evaluations:
                global_iteration_id = evaluation["global_iteration"]
                if iter_id == global_iteration_id and "evaluations" in evaluation:
                    evaluations = evaluation["evaluations"]
                    for learner_id, learner_evaluation in evaluations.items():
                        iteration_performances.append(
                            float(learner_evaluation["test_evaluation"]["metric_values"][metric]))
            if len(iteration_performances) > 0:
                iteration_performance = np.mean(iteration_performances)
                global_model_test_performances.append(iteration_performance)
        global_iterations_index = global_iterations_index[:len(global_model_test_performances)]
        global_iterations_ts = global_iterations_ts[:len(global_model_test_performances)]
        print(global_model_test_performances)
        ax.plot(global_iterations_ts, global_model_test_performances, label=line_label)

        for learner_id, learner_values in learners_task_results.items():
            tasks_metadata = list(learner_values['task_metadata'])
            tasks_training_evaluation = \
                [(t["global_iteration"],
                  float(t["task_evaluation"]["training_evaluation"][-1]["model_evaluation"]["metric_values"][metric]))
                 for t in tasks_metadata]

            tasks_test_evaluation = \
                [(t["global_iteration"],
                  float(t["task_evaluation"]["test_evaluation"][-1]["model_evaluation"]["metric_values"][metric]))
                 for t in tasks_metadata]

            tasks_training_evaluation_sorted = []
            tasks_test_evaluation_sorted = []
            for iter_id, ts in global_iterations_sorted:
                for train_eval in tasks_training_evaluation:
                    if iter_id == train_eval[0]:
                        tasks_training_evaluation_sorted.append(train_eval[1])
                for test_eval in tasks_test_evaluation:
                    if iter_id == test_eval[0]:
                        tasks_test_evaluation_sorted.append(test_eval[1])

            # ax.plot(global_iterations_ts, tasks_test_evaluation_sorted, label=learner_id)

    # 3d model centralized: 2.694456259
    # 2d model centralized: 2.667021116
    ax.plot(global_iterations_ts, len(global_iterations_ts) * [2.895984411239624],
            linestyle="solid", color='crimson',
            linewidth=2, label="Centralized")

    fig.legend()
    return fig


if __name__ == "__main__":
    # plot_rounds_convergence(
    #     "/private/var/tmp/_bazel_Dstrip/6e1f0333b1e46ed6aeb019f1648d0665/execroot/projectmetis/bazel-out/darwin-fastbuild/bin/experiments/keras/fashionmnist.runfiles/projectmetis/experiments/keras/experiment.json",
    #     metric="accuracy")
    # files = glob.glob("/Users/Dstrip/CLionProjects/projectmetis-rc/experiments/execution_logs/brainage/federated/3dmodel/*3D*")
    files = glob.glob("/Users/Dstrip/Downloads/experiment.json")
    fig = plot_rounds_convergence(files, metric="mae")
    pdf_out = "/Users/Dstrip/CLionProjects/projectmetis-rc/experiments/analysis/{}.pdf".format('Policies Convergence')
    pdfpages1 = PdfPages(pdf_out)
    pdfpages1.savefig(figure=fig, bbox_inches='tight')
    pdfpages1.close()
    # plt.show()
