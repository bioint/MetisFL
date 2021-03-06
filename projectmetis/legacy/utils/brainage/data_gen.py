
import math
import nibabel as nib
import numpy as np
import scipy.stats as ss
import os
import pandas as pd
import tensorflow as tf
import random


class Scan_Gen():
    
    def __init__(
        self, batch_size=10, data_config=None, 
        row_offset=0, col_offset=0, depth_offset=0,
        rows=50, cols=50, depth=50, pthreads=1):
        
        self.batch_size  = batch_size 
        self.data_config = data_config
        
        self.row_offset   = row_offset
        self.col_offset   = col_offset
        self.depth_offset = depth_offset
        
        self.rows      = rows
        self.cols      = cols
        self.depth     = depth
        self.channels = len(data_config)
        self.pthreads = pthreads
    

    def parse_table(self, subj_table, req_col=["eid","age_at_scan"]):
        
        # Collect names of required columns
        img_col = []
        for channel in self.data_config:
            img_col.append(channel)
        
        # Allow for convenient pred (No AgeAtScan)
        if not "age_at_scan" in subj_table.columns:
            subj_table["age_at_scan"] = -1
        
        if not set(img_col + req_col).issubset(subj_table.columns):
            print("Error: Missing columns in table!")
            exit(1)
        
        # Combine all image data paths together as one column
        subj_table["message"] = subj_table[img_col].apply(lambda x: ",".join(x), axis=1)
        
        # Remove all irrelevant columns
        subj_table = subj_table[req_col + ["message"]]
        
        return subj_table.values # numpy array


    def create_dataset_wschema(self, filepath, is_training, req_col=["eid", "age_at_scan"]):
        if not os.path.exists(filepath):
            print("Error: Filepath {} does not exist!".format(filepath))
            exit(1)

        subj_table = pd.read_csv(filepath)

        parsed = self.parse_table(subj_table, req_col)

        # Not generalized...
        slice0 = parsed[:, 0].tolist()  # Subject
        slice1 = parsed[:, 1].tolist()  # AgeAtScan
        slice2 = parsed[:, 2].tolist()  # message

        num_rows = len(slice0)
        dataset = tf.data.Dataset.from_tensor_slices((slice0, slice1, slice2))

        if is_training:
            # TODO This is a CRITICAL step. If not shuffled then no generalization!
            dataset = dataset.shuffle(buffer_size=num_rows)

        dataset = dataset.map(
            lambda subj, age, mess: tf.py_func(
                self.load_v2,
                (subj, age, mess), (tf.string, tf.float32, tf.float32, tf.float32)),
            num_parallel_calls=self.pthreads)

        # Bundle tensors together as key-value attributes.
        dataset = tf.data.Dataset.zip(({'subject_id': dataset.map(map_func=lambda w, x, y, z: w),
                                        'age': dataset.map(map_func=lambda w, x, y, z: x),
                                        'dist': dataset.map(map_func=lambda w, x, y, z: y),
                                        'images': dataset.map(map_func=lambda w, x, y, z: z)}))

        return dataset, num_rows


    def get_batch(self, handle, output_types, output_shapes):
        iterator = tf.data.Iterator.from_string_handle(
            handle, output_types, output_shapes)
        
        ids, scans, dists = iterator.get_next()        
        
        ids.set_shape([None, 1])
        scans.set_shape([None, self.rows, self.cols, self.depth, self.channels])
        dists.set_shape([None, 1])	
        
        return ids, scans, dists


    def load_v2(self, subj_id, age, mess):
        # mess is a panda frame; decode will comma split the string
        mess = mess.decode()
        mess = mess.split(",")

        # Add each as a channel
        cpath = mess[0]
        img = nib.load(cpath).get_fdata()
        # print("mean")
        # print(img.mean())
        # print("std")
        # print(img.std())
        img = (img - img.mean()) / img.std()
        # print(cpath)
        scan = np.float32(img[:, :, :, np.newaxis])

        age = np.float32(age)

        # Hard-coded bins(36) for UKBB
        # with mean at age
        x = np.arange(45, 81)

        # Use upper and lower bounds on bins
        # to estimate prob values
        xU, xL = x + 0.05, x - 0.05
        cdfU = ss.norm.cdf(xU, loc=age, scale=5.5)
        cdfL = ss.norm.cdf(xL, loc=age, scale=5.5)
        dist = cdfU - cdfL
        dist = dist / dist.sum()
        dist = np.float32(dist)
        # dist = [np.float32(dist)]

        # Standardize for int subj id
        # String subj are already converted
        if not type(subj_id) is bytes:
            subj_id = str(subj_id).encode("UTF-8")

        return subj_id, [age], dist, scan

