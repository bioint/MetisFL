import random

import pandas as pd
import numpy as np

from collections import defaultdict

np.random.seed(seed=1990)
random.seed(a=1990)

TRAIN_PATH = "../centralized/train.csv"
SIZES = [900, 900, 900, 900, 900, 900, 900, 900]
print(SIZES)
MEANS = [50, 60, 70, 80, 50, 60, 70, 80]
STD   = 7 

data = pd.read_csv(TRAIN_PATH)

# Map site idx to list of dataframes
node_to_data = defaultdict(list)
keep_running = True

# Stop when dataset is empty or when
# all sites are at capacity
while keep_running and len(data) > 0:

    keep_running = False # Stop when all sites reach capacity
    for i in range(len(MEANS)):
        if len(data) == 0:
            break
        if len(node_to_data[str(i)]) < SIZES[i]:
            keep_running = True

        sample = np.random.normal(loc=MEANS[i], scale=STD)
        # Find closest value to sample
        x = data.iloc[(data["age_at_scan"]-sample).abs().argsort()[:1]].iloc[0]
        # Add subj to site set
        node_to_data[str(i)].append(x)
        # Remove subj from dataset
        data = data[data["eid"] != x["eid"]]

# If still remaining data, add them arbitrarily
# uniformly across sites
if len(data) > 0:
    while len(data) > 0:
        for i in range(len(SIZES)):
            if len(data) > 0:
                x = data.iloc[0]
                # Add subj to site set
                node_to_data[str(i)].append(x)
                data = data[data["eid"] != x["eid"]]

for sidx in node_to_data:
    fidx = str(int(sidx) + 1)
    pd.DataFrame(node_to_data[sidx]).to_csv("{}.csv".format(fidx), index=False)
    partition_data = [x['age_at_scan'] for x in node_to_data[sidx]]
    print("Partition ID: ", fidx)
    print("Partition Size: ", len(partition_data))
    print("Partition Data: ", partition_data)
    print("Mean: {}, STD: {}".format(np.mean(partition_data), np.std(partition_data)))

print(len(data))