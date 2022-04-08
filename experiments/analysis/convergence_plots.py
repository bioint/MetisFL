import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

def convert_to_timestamp(t):
    return datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ').timestamp()


def plot_rounds_convergence(files, metric="accuracy"):

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for file in files:

        line_label = "global model"
        if "uniform_iid" in file:
            line_label = "Uni-IID"
        if "uniform_noniid" in file:
            line_label = "Uni-NonIID"
        if "skewed_iid" in file:
            line_label = "Ske-IID"
        if "skewed_noniid" in file:
            line_label = "Ske-NonIID"

        print(file, flush=True)
        json_data = json.load(open(file, "r"))
        federation_runtime_metadata = json_data["federation_runtime_metadata"]["metadata"]
        completed_global_iterations = \
            [(iteration["global_iteration"], convert_to_timestamp(iteration["completed_at"]))
             for iteration in federation_runtime_metadata if "completed_at" in iteration]
        global_iterations_sorted = sorted(completed_global_iterations, key=lambda x: x[1])  # sort by timestamp
        initial_timestamp = global_iterations_sorted[0][1]
        print([x[1]-initial_timestamp for x in global_iterations_sorted[1:]], flush=True)
        # default is seconds - /60 to minutes
        global_iterations = [(x[0], x[1]-initial_timestamp) for x in global_iterations_sorted]
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
                        iteration_performances.append(float(learner_evaluation["test_evaluation"]["metric_values"][metric]))
            if len(iteration_performances) > 0:
                iteration_performance = np.mean(iteration_performances)
                global_model_test_performances.append(iteration_performance)
        global_iterations_index = global_iterations_index[:len(global_model_test_performances)]
        global_iterations_ts = global_iterations_ts[:len(global_model_test_performances)]
        print(global_model_test_performances)
        ax.plot(global_iterations_index, global_model_test_performances, label=line_label)

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

    ax.plot(global_iterations_ts, len(global_iterations_ts) * [2.895984411239624],
            linestyle="solid", color='crimson',
            linewidth=2, label="Centralized")

    fig.legend()
    return fig


if __name__ == "__main__":
    # plot_rounds_convergence(
    #     "/private/var/tmp/_bazel_Dstrip/6e1f0333b1e46ed6aeb019f1648d0665/execroot/projectmetis/bazel-out/darwin-fastbuild/bin/experiments/keras/fashionmnist.runfiles/projectmetis/experiments/keras/experiment.json",
    #     metric="accuracy")
    # files = glob.glob("/Users/Dstrip/CLionProjects/projectmetis-rc/experiments/execution_logs/brainage*")
    files = glob.glob("/Users/Dstrip/Downloads/experiment.json")
    fig = plot_rounds_convergence(files, metric="accuracy")
    pdf_out = "/Users/Dstrip/CLionProjects/projectmetis-rc/experiments/analysis/{}.pdf".format('Policies Convergence')
    pdfpages1 = PdfPages(pdf_out)
    pdfpages1.savefig(figure=fig, bbox_inches='tight')
    pdfpages1.close()
    # plt.show()
