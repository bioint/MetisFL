import numpy as np
import pandas as pd
import glob


AGE_COL_NAME = "age_at_scan"
BASE_PATH = "/Users/Dstrip/PycharmProjects/ProjectMetis-FederatedNeuroImaging/projectmetis/resources/config/experiments_configs/brainage/ukbbdata/uniform_datasize_iid_x8clients/without_validation/*.csv"
OUT_PATH = "./train_{}_corrupted_labels.csv"

for idx, filename in enumerate(glob.glob(BASE_PATH)):

	data = pd.read_csv(filename)
	min_age, max_age = min(data[AGE_COL_NAME]), max(data[AGE_COL_NAME])

	# Sample uniformly at random
	data[AGE_COL_NAME] = np.random.uniform(min_age, max_age, size=len(data))

	print(idx+1, filename)
	data.to_csv(OUT_PATH.format(idx+1), index=False)
