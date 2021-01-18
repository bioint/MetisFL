import random

import federation.fed_cluster_env as fed_cluster_env
import numpy as np
import tensorflow as tf

from utils.objectdetection.imgdata_client import ImgDatasetLoader


np.random.seed(seed=1990)
random.seed(a=1990)
tf.set_random_seed(seed=1990)

def partition_data():

	img_dataset_loader = ImgDatasetLoader(cifar10_loader=True, train_examples=50000, test_examples=0)
	img_dataset_loader.load_image_datasets()
	partitioned_training_data = img_dataset_loader.partition_training_data(partitions_num=10, MLSYS_REBUTTAL_REVIEWER2=True)
	print(partitioned_training_data)


if __name__=="__main__":
	partition_data()
