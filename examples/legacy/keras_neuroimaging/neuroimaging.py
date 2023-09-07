import argparse

import cloudpickle
import collections
import json
import os
import random
import scipy.stats

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

from alzheimers_disease_cnns import AlzheimersDisease2DCNN, AlzheimersDisease3DCNN
from brainage_cnns import BrainAge2DCNN, BrainAge3DCNN

from metisfl.driver.driver import DriverSession
from metisfl.models.model_dataset import ModelDatasetClassification, ModelDatasetRegression
from metisfl.common.fedenv_parser import FederationEnvironment

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SCRIPT_CWD = os.path.dirname(__file__)
print("Script current working directory: ", SCRIPT_CWD, flush=True)

seed_num = 7
random.seed(seed_num)
np.random.seed(seed_num)
tf.random.set_seed(seed_num)


class TFDatasetUtils(object):

    @classmethod
    def int64_feature(cls, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @classmethod
    def float_feature(cls, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @classmethod
    def bytes_feature(cls, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @classmethod
    def generate_tffeature(cls, dataset_records):
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
                feature[k] = cls._bytes_feature(
                    tf.compat.as_bytes(chunk[k_idx].flatten().tostring()))
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
            feature_description[attr] = tf.io.FixedLenFeature(
                shape=[], dtype=tf.string)

        deserialized_example = tf.io.parse_single_example(
            serialized=example_proto, features=feature_description)
        record = []
        for attr in schema_attributes_positioned:
            attr_restored = tf.io.decode_raw(
                deserialized_example[attr], example_schema[attr])
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
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            # Serialize the example to a string
            serialized = example.SerializeToString()
            # Write the serialized object to the file
            tf_record_writer.write(serialized)
        # Close file writer
        tf_record_writer.close()

        return tfrecords_schema


class MRIScanGen(object):

    def __init__(self, filepath, image_column, label_column,
                 rows=91, cols=109, depth=91, channels=1):

        if not os.path.exists(filepath):
            print("Error: Filepath {} does not exist!".format(filepath))
            exit(1)

        self.filepath = filepath

        print(self.filepath)
        self.tfrecord_output = self.filepath + ".tfrecord"
        self.tfrecord_schema_output = self.filepath + ".tfrecord.schema"

        self.image_column = image_column
        self.label_column = label_column
        self.rows = rows
        self.cols = cols
        self.depth = depth
        self.channels = channels

    def parse_csv_table(self, subj_table):

        if not set([self.image_column] + [self.label_column]).issubset(subj_table.columns):
            print("Error: Missing columns in table!")
            exit(1)

        # Remove all irrelevant columns
        subj_table = subj_table[[self.image_column] + [self.label_column]]

        return subj_table

    def generate_tfrecord(self):
        """
        If tfrecord already exists, then just return the tfrecord, else parse
        the .csv file and generate a new tfrecord.
        :return:
        """
        if os.path.exists(self.tfrecord_output) \
                and os.path.exists(self.tfrecord_schema_output):
            tfrecord_schema = cloudpickle.load(
                file=open(self.tfrecord_schema_output, "rb"))
        else:
            subj_table = pd.read_csv(self.filepath)
            subj_table = self.parse_csv_table(subj_table)

            images = subj_table.values[:, 0].tolist()
            labels = subj_table.values[:, 1].tolist()

            parsed_images = []
            parsed_labels = []
            for si, sl in zip(images, labels):
                image, label = self.load_v1(si, sl)
                parsed_images.append(image)
                parsed_labels.append(label)

            final_mappings = collections.OrderedDict()
            final_mappings[self.image_column] = np.array(parsed_images)
            final_mappings[self.label_column] = np.array(labels)

            tfrecord_schema = TFDatasetUtils.serialize_to_tfrecords(
                final_mappings, self.tfrecord_output)
            cloudpickle.dump(obj=tfrecord_schema, file=open(
                self.tfrecord_schema_output, "wb+"))

        return tfrecord_schema

    def load_v1(self, scan_path, label):
        img = nib.load(scan_path).get_fdata()
        # Normalize image.
        img_min = np.amax(img)
        img_max = np.amin(img)
        img = (img - img_min) / (img_max - img_min)
        img = img.astype("float32")
        # Standardize image.
        # img = (img - img.mean()) / img.std()
        # scan = np.float32(img[:, :, :, np.newaxis]) \
        #     .reshape([self.rows, self.cols, self.depth, self.channels])
        label = float(label)
        return img, label

    def process_record(self, image, label):
        # The label is the assigned label to the MRI scan.
        # This could be the age of the scan (regression) or the AD value (classification/binary).
        image = tf.reshape(
            image, [self.rows, self.cols, self.depth, self.channels])
        label = tf.squeeze(label)
        return image, label

    def load_dataset(self, tfrecord_schema):
        # automatically interleaves reads from multiple files
        dataset = tf.data.TFRecordDataset(self.tfrecord_output)
        dataset = dataset.map(map_func=lambda x: TFDatasetUtils
                              .deserialize_single_tfrecord_example(example_proto=x, example_schema=tfrecord_schema),
                              num_parallel_calls=3)
        dataset = dataset.map(map_func=lambda x, y: self.process_record(x, y))
        return dataset

    def get_dataset(self):
        tfrecord_schema = self.generate_tfrecord()
        dataset = self.load_dataset(tfrecord_schema)
        return dataset


class ModelDatasetHelper(object):

    def __init__(self, image_column, label_column):
        self.image_column = image_column
        self.label_column = label_column

    def regression_task_dataset_recipe_fn(self, dataset_fp):
        dataset = MRIScanGen(filepath=dataset_fp,
                             image_column=self.image_column,
                             label_column=self.label_column).get_dataset()
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
            x=dataset, size=len(ages),
            min_val=np.min(ages), max_val=np.max(ages),
            mean_val=np.mean(ages), median_val=np.median(ages),
            mode_val=mode_val, stddev_val=np.std(ages))
        return model_dataset

    def classification_task_dataset_recipe_fn(self, dataset_fp):
        dataset = MRIScanGen(filepath=dataset_fp,
                             image_column=self.image_column,
                             label_column=self.label_column).get_dataset()
        data_iter = iter(dataset)
        classes = []
        for d in data_iter:
            classes.append(int(d[1]))
        records_num = len(classes)
        examples_per_class = {cid: 0 for cid in set(classes)}
        for cid in classes:
            examples_per_class[cid] += 1
        model_dataset = ModelDatasetClassification(
            x=dataset, size=records_num,
            examples_per_class=examples_per_class)
        return model_dataset


if __name__ == "__main__":

    default_neuroimaging_task = "brainage"
    default_neuroimaging_task_model = "3dcnn"
    default_federation_environment_config_fp = os.path.join(
        SCRIPT_CWD, "../config/brainage/brainage_test_localhost_synchronous.yaml")
    default_mri_scans_csv_mapping = os.path.join(
        SCRIPT_CWD, "datasets/ukbb/ukbb_datapaths_absolute.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--neuroimaging_task",
                        default=default_neuroimaging_task,
                        help="Either `brainage` OR `alzheimers`.")
    parser.add_argument("--neuroimaging_task_model",
                        default=default_neuroimaging_task_model,
                        help="Either `3dcnn` OR `2dcnn`.")
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
    train_dataset_recipe_fp_pkl = "{}/model_train_dataset_ops.pkl".format(
        metis_filepath_prefix)
    validation_dataset_recipe_fp_pkl = "{}/model_validation_dataset_ops.pkl".format(
        metis_filepath_prefix)
    test_dataset_recipe_fp_pkl = "{}/model_test_dataset_ops.pkl".format(
        metis_filepath_prefix)

    batch_size = FederationEnvironment(
        args.federation_environment_config_fp).local_model_config.batch_size
    if args.neuroimaging_task == "brainage":
        volume_attr, label_attr = "9dof_2mm_vol", "age_at_scan"
        dataset_recipe_fn = ModelDatasetHelper(volume_attr, label_attr) \
            .regression_task_dataset_recipe_fn
        if args.neuroimaging_task_model == "3dcnn":
            nn_model = BrainAge3DCNN(batch_size=batch_size).get_model()
        elif args.neuroimaging_task_model == "2dcnn":
            nn_model = BrainAge2DCNN(batch_size=batch_size).get_model()
        else:
            raise RuntimeError("Unknown error.")
    elif args.neuroimaging_task == "alzheimers":
        volume_attr, label_attr = "volume", "label"
        dataset_recipe_fn = ModelDatasetHelper(volume_attr, label_attr) \
            .classification_task_dataset_recipe_fn
        if args.neuroimaging_task_model == "3dcnn":
            nn_model = AlzheimersDisease3DCNN().get_model()
        elif args.neuroimaging_task_model == "2dcnn":
            nn_model = AlzheimersDisease2DCNN().get_model()
        else:
            raise RuntimeError("Unknown error.")
    else:
        raise RuntimeError("Unknown task.")

    # Load dummy data for model initialization purposes.
    dummy_dataset = MRIScanGen(
        filepath=args.dummy_scans_csv_mapping_fp,
        image_column=volume_attr,
        label_column=label_attr) \
        .get_dataset()

    nn_model.summary()
    nn_model.evaluate(x=dummy_dataset.batch(1))
    nn_model.save(model_definition_dir)

    cloudpickle.dump(obj=dataset_recipe_fn, file=open(
        train_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(
        test_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(
        validation_dataset_recipe_fp_pkl, "wb+"))

    driver_session = DriverSession(args.federation_environment_config_fp,
                                   model=nn_model,
                                   train_dataset_recipe_fn=dataset_recipe_fn,
                                   validation_dataset_recipe_fn=dataset_recipe_fn,
                                   test_dataset_recipe_fn=dataset_recipe_fn)
    driver_session.initialize_federation()
    driver_session.monitor_federation()
    driver_session.shutdown_federation()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(SCRIPT_CWD, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
