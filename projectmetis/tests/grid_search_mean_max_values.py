from utils.logging.logs_processing_ops import LogsProcessingUtil
from collections import OrderedDict
import numpy as np
import glob
import os

scriptDirectory = os.path.dirname(os.path.realpath(__file__))
grid_search_directory = scriptDirectory + "/../resources/logs/testing_producibility/cifar10_FedDiIFVLVal_InfiniteFVL_Val_005_grid_search_v2"
grid_search_files = glob.glob(grid_search_directory + "/*.json")
_, _, asynchronous = LogsProcessingUtil.load_files_to_dict(grid_search_files, federation_policies=True)

all_collections = {}
for idx, async_collection in enumerate(asynchronous):
	asynchronous_wall_time_with_scores = LogsProcessingUtil.compute_wall_clock_time_with_test_score_across_federation(async_collection,
																													  metric="accuracy",
																													  synchronous_policy=False)
	current_filename = grid_search_files[idx]
	line_labels, filename_labels = LogsProcessingUtil.get_labels_from_filename(current_filename, synchronous=False)
	asynchronous_wall_time = [x[0] for x in asynchronous_wall_time_with_scores]
	asynchronous_wall_time_scores = [x[1] for x in asynchronous_wall_time_with_scores]
	top_five = sorted(asynchronous_wall_time_scores)[-5:]
	# top_value = np.round(np.mean(top_five), 4)
	top_value = np.round(top_five[-1], 4)
	dict_key = (current_filename.split('InfiniteFVL.')[1]).split('.SGDWithMomentum075')[0]
	all_collections[dict_key] = top_value

all_collections = OrderedDict(sorted(all_collections.items(), key=lambda x: x[1]))
for collection in all_collections:
	print(collection, all_collections[collection])
