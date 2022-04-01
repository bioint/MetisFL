import argparse
import cloudpickle
import collections
import json
import os
import scipy.stats

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

from experiments.keras.models.alzheimer_disease_cnns import AlzheimerDisease2DCNN, AlzheimerDisease3DCNN
from experiments.keras.models.brainage_cnns import BrainAge2DCNN, BrainAge3DCNN
from projectmetis.python.driver.driver_session import DriverSession
from projectmetis.python.models.model_dataset import ModelDatasetRegression

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TFDatasetUtils(object):

    @classmethod
    def _int64_feature(cls, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @classmethod
    def _float_feature(cls, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @classmethod
    def _bytes_feature(cls, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod
    def _generate_tffeature(cls, dataset_records):
        # Loop over the schema keys.
        record_keys = dataset_records.keys()
        # We split the input arrays in one-to-one examples.
        records_chunks = list()
        for k in record_keys:
            np_array = dataset_records[k]
            records_chunks.append(np.array_split(np_array, np_array.shape[0]))

        # Per example generator yield.
        for chunk in zip(*records_chunks):
            feature = {}
            # Convert every attribute to tf compatible feature.
            for k_idx, k in enumerate(record_keys):
                feature[k] = cls._bytes_feature(tf.compat.as_bytes(chunk[k_idx].flatten().tostring()))
            yield feature

    @classmethod
    def deserialize_single_tfrecord_example(cls, example_proto: tf.Tensor, example_schema: dict):
        """
        If the input schema is already ordered then do not change keys order
        and use this sequence to deserialize the records. Else sort the keys
        by name and use the alphabetical sequence to deserialize.
        :param example_proto:
        :param example_schema:
        :return:
        """
        assert isinstance(example_proto, tf.Tensor)
        assert isinstance(example_schema, dict)

        if not isinstance(example_schema, collections.OrderedDict):
            schema_attributes_positioned = list(sorted(example_schema.keys()))
        else:
            schema_attributes_positioned = list(example_schema.keys())

        feature_description = dict()
        for attr in schema_attributes_positioned:
            feature_description[attr] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)

        deserialized_example = tf.io.parse_single_example(
            serialized=example_proto, features=feature_description)
        record = []
        for attr in schema_attributes_positioned:
            attr_restored = tf.io.decode_raw(deserialized_example[attr], example_schema[attr])
            record.append(attr_restored)

        return record

    @classmethod
    def serialize_to_tfrecords(cls, dataset_records_mappings: dict, output_filename: str):
        """
        The `dataset_records_mappings` is a dictionary with format:
            {"key1" -> np.ndarray(), "key2" -> np.ndarray(), etc...}
        Using this dict we zip ndarrays rows and we serialize them as tfrecords
        to the output_filename. The schema (attributes) of the serialized tfrecords
        is based on the dictionary keys. The order of the keys in the input dictionary is
        preserved and is used to serialize to tfrecords.
        :param dataset_records_mappings:
        :param output_filename:
        :return:
        """
        assert isinstance(dataset_records_mappings, dict)
        for val in dataset_records_mappings.values():
            assert isinstance(val, np.ndarray)

        # Attributes tf.data_type is returned in alphabetical order
        tfrecords_schema = collections.OrderedDict(
            {attr: tf.as_dtype(val.dtype.name) for attr, val in dataset_records_mappings.items()})

        # Open file writer
        tf_record_writer = tf.io.TFRecordWriter(output_filename)
        # Iterate over dataset's features generator
        for feature in cls._generate_tffeature(dataset_records_mappings):
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize the example to a string
            serialized = example.SerializeToString()
            # Write the serialized object to the file
            tf_record_writer.write(serialized)
        # Close file writer
        tf_record_writer.close()

        return tfrecords_schema


class MRIScanGen(object):

    def __init__(self, filepath, data_config, req_col,
                 rows=91, cols=109, depth=91, channels=1):

        if not os.path.exists(filepath):
            print("Error: Filepath {} does not exist!".format(filepath))
            exit(1)

        self.filepath = filepath
        self.tfrecord_output = self.filepath + ".tfrecord"
        self.tfrecord_schema_output = self.filepath + ".tfrecord.schema"

        self.data_config = [data_config]
        self.req_col = [req_col]
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.channels = channels

    def parse_csv_table(self, subj_table):
        # Collect names of required columns
        img_col = []
        for channel in self.data_config:
            img_col.append(channel)

        # Allow for convenient pred (No AgeAtScan)
        if "age_at_scan" not in subj_table.columns:
            subj_table["age_at_scan"] = -1

        if not set(img_col + self.req_col).issubset(subj_table.columns):
            print("Error: Missing columns in table!")
            exit(1)

        # Combine all image data paths together as one column
        subj_table["scan_path"] = subj_table[img_col].apply(lambda x: ",".join(x), axis=1)

        # Remove all irrelevant columns
        subj_table = subj_table[self.req_col + ["scan_path"]]

        return subj_table

    def generate_tfrecord(self):

        if os.path.exists(self.tfrecord_output) \
                and os.path.exists(self.tfrecord_schema_output):
            tfrecord_schema = cloudpickle.load(file=open(self.tfrecord_schema_output, "rb"))
        else:
            subj_table = pd.read_csv(self.filepath)
            data_mappings = self.parse_csv_table(subj_table)

            ages = data_mappings.values[:, 0].tolist()  # age_at_scan
            scan_paths = data_mappings.values[:, 1].tolist()  # 9dof_2mm_vol.nii scan path

            parsed_scans = []
            for s0, s1 in zip(ages, scan_paths):
                scan, age = self.load_v1(s0, s1)
                parsed_scans.append(scan)
            parsed_scans_np = np.array(parsed_scans, dtype=np.float32)

            final_mappings = collections.OrderedDict()
            final_mappings["scan_images"] = parsed_scans_np
            for col in self.req_col:
                final_mappings[col] = data_mappings[col].values

            tfrecord_schema = TFDatasetUtils.serialize_to_tfrecords(
                final_mappings, self.tfrecord_output)
            cloudpickle.dump(obj=tfrecord_schema, file=open(self.tfrecord_schema_output, "wb+"))

        return tfrecord_schema

    def load_v1(self, age, scan_path):
        img = nib.load(scan_path).get_fdata()
        img = (img - img.mean()) / img.std()
        # scan = np.float32(img[:, :, :, np.newaxis]) \
        #     .reshape([self.rows, self.cols, self.depth, self.channels])
        age = float(age)
        return img, age

    def process_record(self, image, age):
        image = tf.reshape(image, [self.rows, self.cols, self.depth, self.channels])
        age = tf.squeeze(age)
        return image, age

    def load_dataset(self, tfrecord_schema):
        dataset = tf.data.TFRecordDataset(self.tfrecord_output)  # automatically interleaves reads from multiple files
        dataset = dataset.map(map_func=lambda x: TFDatasetUtils
                              .deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecord_schema),
                              num_parallel_calls=3)
        dataset = dataset.map(map_func=lambda x, y: self.process_record(x, y))
        return dataset

    def get_dataset(self):
        tfrecord_schema = self.generate_tfrecord()
        dataset = self.load_dataset(tfrecord_schema)
        return dataset


if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    default_neuroimaging_task = "brainage"
    default_neuroimaging_task_model = "3dcnn"
    default_federation_environment_config_fp = os.path.join(
        script_cwd, "../federation_environments_config/brainage/brainage_test_localhost_synchronous.yaml")
    default_mri_scans_csv_mapping = os.path.join(
        script_cwd, "datasets/ukbb/ukbb_datapaths_absolute.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--neuroimaging_task",
                        default=default_neuroimaging_task)
    parser.add_argument("--neuroimaging_task_model",
                        default=default_neuroimaging_task_model)
    parser.add_argument("--federation_environment_config_fp",
                        default=default_federation_environment_config_fp)
    parser.add_argument("--dummy_scans_csv_mapping_fp",
                        default=default_mri_scans_csv_mapping)
    args = parser.parse_args()
    print(args, flush=True)

    nn_engine = "keras"
    metis_filepath_prefix = "/tmp/metis/model"
    if not os.path.exists(metis_filepath_prefix):
        os.makedirs(metis_filepath_prefix)

    model_definition_dir = "{}/model_definition".format(metis_filepath_prefix)
    train_dataset_recipe_fp_pkl = "{}/model_train_dataset_ops.pkl".format(metis_filepath_prefix)
    validation_dataset_recipe_fp_pkl = "{}/model_validation_dataset_ops.pkl".format(metis_filepath_prefix)
    test_dataset_recipe_fp_pkl = "{}/model_test_dataset_ops.pkl".format(metis_filepath_prefix)

    if args.neuroimaging_task == "brainage":
        volume_attr, label_attr = "9dof_2mm_vol",  "age_at_scan"
        if args.neuroimaging_task_model == "3dcnn":
            nn_model = BrainAge3DCNN().get_model()
        elif args.neuroimaging_task_model == "2dcnn":
            nn_model = BrainAge2DCNN().get_model()
        else:
            raise RuntimeError("Unknown error.")
    elif args.neuroimaging_task == "alzheimer":
        volume_attr, label_attr = "volume", "label"
        if args.neuroimaging_task_model == "3dcnn":
            nn_model = AlzheimerDisease3DCNN().get_model()
        elif args.neuroimaging_task_model == "2dcnn":
            nn_model = AlzheimerDisease2DCNN().get_model()
        else:
            raise RuntimeError("Unknown error.")
    else:
        raise RuntimeError("Unknown task.")

    # Load dummy data for model initialization purposes.
    volume_attr, label_attr = "9dof_2mm_vol",  "age_at_scan"
    dummy_dataset = MRIScanGen(
        filepath=args.dummy_scans_csv_mapping_fp,
        data_config=volume_attr,
        req_col=label_attr) \
        .get_dataset()

    nn_model.summary()
    nn_model.evaluate(x=dummy_dataset.batch(1))
    nn_model.save(model_definition_dir)

    def dataset_recipe_fn(dataset_fp):
        dataset = MRIScanGen(
            filepath=dataset_fp,
            data_config=volume_attr,
            req_col=label_attr) \
            .get_dataset()

        data_iter = iter(dataset)
        ages = []
        for d in data_iter:
            ages.append(float(d[1]))
        mode_values, mode_counts = scipy.stats.mode(ages)
        if np.all((mode_counts == 1)):
            mode_val = np.max(ages)
        else:
            mode_val = mode_values[0]
        model_dataset = ModelDatasetRegression(
            dataset=dataset, size=len(ages),
            min_val=np.min(ages), max_val=np.max(ages),
            mean_val=np.mean(ages), median_val=np.median(ages),
            mode_val=mode_val, stddev_val=np.std(ages))
        return model_dataset

    cloudpickle.dump(obj=dataset_recipe_fn, file=open(train_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(test_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(validation_dataset_recipe_fp_pkl, "wb+"))

    driver_session = DriverSession(args.federation_environment_config_fp, nn_engine,
                                   model_definition_dir=model_definition_dir,
                                   train_dataset_recipe_fp=train_dataset_recipe_fp_pkl,
                                   validation_dataset_recipe_fp=validation_dataset_recipe_fp_pkl,
                                   test_dataset_recipe_fp=test_dataset_recipe_fp_pkl)
    driver_session.initialize_federation(model_weights=nn_model.get_weights())
    driver_session.monitor_federation()
    driver_session.shutdown_federation()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
