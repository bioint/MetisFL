import cloudpickle
import os
import scipy.stats

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.keras.models.brainage_3dcnn import BrainAge3DCNN
from projectmetis.python.driver.driver_session import DriverSession
from projectmetis.python.models.model_dataset import ModelDatasetRegression


class MRIScanGen(object):

    def __init__(self, working_dir="", data_config=["9dof_2mm_vol"],
                 rows=91, cols=109, depth=91, channels=1):

        self.working_dir = working_dir
        self.data_config = data_config

        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.channels = channels

    def _parse_csv_table(self, subj_table, req_col=["eid", "age_at_scan"]):

        # Collect names of required columns
        img_col = []
        for channel in self.data_config:
            img_col.append(channel)

        # Allow for convenient pred (No AgeAtScan)
        if "age_at_scan" not in subj_table.columns:
            subj_table["age_at_scan"] = -1

        if not set(img_col + req_col).issubset(subj_table.columns):
            print("Error: Missing columns in table!")
            exit(1)

        # Combine all image data paths together as one column
        subj_table["message"] = subj_table[img_col].apply(lambda x: ",".join(x), axis=1)

        # Remove all irrelevant columns
        subj_table = subj_table[req_col + ["message"]]

        return subj_table.values  # numpy array

    def parse_dataset(self, filepath, req_col=["eid", "age_at_scan"]):
        if not os.path.exists(filepath):
            print("Error: Filepath {} does not exist!".format(filepath))
            exit(1)

        subj_table = pd.read_csv(filepath, encoding_errors="ignore")

        parsed = self._parse_csv_table(subj_table, req_col)

        slice0 = parsed[:, 0].tolist()  # Subject
        slice1 = parsed[:, 1].tolist()  # AgeAtScan
        slice2 = parsed[:, 2].tolist()  # .nii Scans

        parsed_scans = []
        parsed_ages = []
        for s0, s1, s2 in zip(slice0, slice1, slice2):
            subj_id = s0
            scan, age = self.load_v1(s1, s2)
            parsed_scans.append(scan)
            parsed_ages.append(age)

        parsed_scans_np = np.array(parsed_scans)
        parsed_ages_np = np.array(parsed_ages)
        return parsed_scans_np, parsed_ages_np

    def load_v1(self, age, scan_path):
        scan_path = os.path.join(self.working_dir, scan_path)
        img = nib.load(scan_path).get_fdata()
        img = (img - img.mean()) / img.std()
        scan = np.float32(img[:, :, :, np.newaxis])\
            .reshape([self.rows, self.cols, self.depth, self.channels])
        age = np.float32(age)

        return scan, age


if __name__ == "__main__":

    nn_engine = "keras"
    metis_filepath_prefix = "/tmp/metis/model"
    if not os.path.exists(metis_filepath_prefix):
        os.makedirs(metis_filepath_prefix)

    model_definition_dir = "{}/model_definition".format(metis_filepath_prefix)
    train_dataset_filepath = "{}/model_train_dataset.npz".format(metis_filepath_prefix)
    validation_dataset_filepath = "{}/model_validation_dataset.npz".format(metis_filepath_prefix)
    test_dataset_filepath = "{}/model_test_dataset.npz".format(metis_filepath_prefix)
    train_dataset_recipe_fp_pkl = "{}/model_train_dataset_ops.pkl".format(metis_filepath_prefix)
    validation_dataset_recipe_fp_pkl = "{}/model_validation_dataset_ops.pkl".format(metis_filepath_prefix)
    test_dataset_recipe_fp_pkl = "{}/model_test_dataset_ops.pkl".format(metis_filepath_prefix)

    brainage_model = BrainAge3DCNN().get_model()

    """ Load the data. """
    dirname = os.path.dirname(__file__)
    mri_scans_csv_mapping = os.path.join(
        dirname, "datasets/brainage/brainage_datapaths_relative.csv")
    x_train, y_train = MRIScanGen(working_dir=dirname, data_config=["9dof_2mm_vol"])\
        .parse_dataset(filepath=mri_scans_csv_mapping)
    x_valid, y_valid = MRIScanGen(working_dir=dirname, data_config=["9dof_2mm_vol"]) \
        .parse_dataset(filepath=mri_scans_csv_mapping)
    x_test, y_test = MRIScanGen(working_dir=dirname, data_config=["9dof_2mm_vol"]) \
        .parse_dataset(filepath=mri_scans_csv_mapping)

    # Save data.
    np.savez(train_dataset_filepath, x=x_train, y=y_train)
    np.savez(validation_dataset_filepath, x=x_valid, y=y_valid)
    np.savez(test_dataset_filepath, x=x_test, y=y_test)

    nn_model = brainage_model
    # Perform an .evaluation() step to initialize all Keras 'hidden' states, else model.save() will not save the model
    # properly and any subsequent fit step will never train the model properly.
    nn_model.evaluate(x=np.random.random(x_train[0:1].shape), y=np.random.random(y_train[0:1].shape), verbose=False)
    nn_model.save(model_definition_dir)

    def dataset_recipe_fn(dataset_fp):
        x, y = MRIScanGen(data_config=["9dof_2mm_vol"])\
            .parse_dataset(filepath=dataset_fp)

        mode_values, mode_counts = scipy.stats.mode(y_train)
        if np.all((mode_counts == 1)):
            mode_val = np.max(y_train)
        else:
            mode_val = mode_values[0]
        model_dataset = ModelDatasetRegression(
            x=x, y=y, size=y.size,
            min_val=np.min(y), max_val=np.max(y),
            mean_val=np.mean(y), median_val=np.median(y),
            mode_val=mode_val, stddev_val=np.std(y))
        return model_dataset

    cloudpickle.dump(obj=dataset_recipe_fn, file=open(train_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(test_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(validation_dataset_recipe_fp_pkl, "wb+"))

    dirname = os.path.dirname(__file__)
    federation_environment_config_fp = os.path.join(
        dirname, "../federation_environments_config/test_localhost.yaml")
    driver_session = DriverSession(federation_environment_config_fp, nn_engine,
                                   model_definition_dir=model_definition_dir,
                                   train_dataset_recipe_fp=train_dataset_recipe_fp_pkl,
                                   validation_dataset_recipe_fp=validation_dataset_recipe_fp_pkl,
                                   test_dataset_recipe_fp=test_dataset_recipe_fp_pkl)
    driver_session.initialize_federation()
    driver_session.monitor_federation()
