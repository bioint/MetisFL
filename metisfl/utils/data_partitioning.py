import random

import numpy as np

from metisfl.utils.logger import MetisLogger


class DataPartitioning(object):

    def __init__(self, x_train, y_train, partitions_num, seed=1990):
        self.x_train = x_train
        self.y_train = y_train
        self.partitions_num = partitions_num
        self.seed = seed
        self.iid = False
        self.non_iid = False
        self.classes_per_client = 0
        self.examples_per_client = []

    def iid_partition(self):
        idx = list(range(len(self.x_train)))
        # Seed value needs to be set every time a random operation is invoked.
        random.seed(self.seed)
        random.shuffle(idx)
        x_train_randomized = self.x_train[idx]
        y_train_randomized = self.y_train[idx]

        chunk_size = int(len(self.x_train) / self.partitions_num)
        x_chunks, y_chunks = [], []
        for i in range(self.partitions_num):
            x_chunks.append(x_train_randomized[idx[i * chunk_size:(i + 1) * chunk_size]])
            y_chunks.append(y_train_randomized[idx[i * chunk_size:(i + 1) * chunk_size]])
        x_chunks = np.array(x_chunks)
        y_chunks = np.array(y_chunks)
        MetisLogger.info("Chunk size {}, {}".format(x_chunks.shape, y_chunks.shape))

        # set partition object specifications
        self.iid = True
        self.classes_per_client = np.unique(y_chunks).size
        self.examples_per_client = [y_c.size for y_c in y_chunks]

        return x_chunks, y_chunks

    def non_iid_partition(self, classes_per_partition=2):

        """ If the y-data points contain numpy arrays, we need to convert it
        to a list of values in order for the set/hash to take effect during
        partitioning by the y-axis. """
        y_are_ndarrays = False
        if isinstance(self.y_train[0], np.ndarray):
            y_are_ndarrays = True
            y_converted_values = []
            for v in self.y_train:
                y_converted_values.extend(v.tolist())
        else:
            y_converted_values = self.y_train

        sorted_data = sorted(zip(self.x_train, y_converted_values), key=lambda pair: pair[1])
        x_train_sorted = [x for x, y in sorted_data]
        y_train_sorted = [y for x, y in sorted_data]

        # The number of chunks depends on the number of partitions and number of classes.
        # If we want one class per client then we split the data into #partitions == #clients
        # else, if we want k-classes per client then we need to split the data into #partitions * #k-classes
        chunk_size = int(len(self.x_train) / (self.partitions_num * classes_per_partition))

        x_chunks = [x_train_sorted[i:i + chunk_size] for i in range(0, len(x_train_sorted), chunk_size)]
        y_chunks = [y_train_sorted[i:i + chunk_size] for i in range(0, len(y_train_sorted), chunk_size)]

        x_chunks_all_clients, y_chunks_all_clients = [], []
        assigned_chunks = dict()
        for pidx in range(self.partitions_num):
            assigned_classes = 0
            indexes_to_remove = []
            x_chunks_single_client, y_chunks_single_client = [], []
            for chunk_idx, y_chunk in enumerate(y_chunks):
                if assigned_classes < classes_per_partition:
                    # Make sure that there is no overlap between classes.
                    if len(set(y_chunks_single_client).intersection(set(y_chunk))) == 0:
                        y_chunks_single_client.extend(y_chunk)
                        x_chunks_single_client.extend(x_chunks[chunk_idx])
                        assigned_chunks[chunk_idx] = True
                        assigned_classes += 1
                        indexes_to_remove.append(chunk_idx)
                else:
                    # If limit of assigned classes is reached then exit.
                    break
            x_chunks_all_clients.append(x_chunks_single_client)
            y_chunks_all_clients.append(y_chunks_single_client)

            for position, idx in enumerate(indexes_to_remove):
                del x_chunks[idx - position]
                del y_chunks[idx - position]

        x_chunks_final = np.array(x_chunks_all_clients)

        """ Bring the format of the y-values back to their original numpy array format. """
        if y_are_ndarrays:
            for idx, y_chunk in enumerate(y_chunks_all_clients):
                y_chunks_all_clients[idx] = [np.array(y) for y in y_chunk]

        y_chunks_final = np.array(y_chunks_all_clients)
        MetisLogger.info("Chunk size {}. X-attribute shape: {}, Y-attribute shape: {}".format(
            chunk_size, x_chunks_final.shape, y_chunks_final.shape))
        remaining = len(y_chunks)
        MetisLogger.info("Remaining unassigned data points: {}".format(len(y_chunks)))
        if remaining > 0:
            MetisLogger.fatal("Not all training data have been assigned.")

        # set partition object specifications
        self.non_iid = True
        self.classes_per_client = classes_per_partition
        self.examples_per_client = [y_c.size for y_c in y_chunks_final]

        return x_chunks_final, y_chunks_final

    def dirichlet_based_partition(self, a):
        pass

    def to_json_representation(self):
        return {'iid': self.iid,
                'non_iid': self.non_iid,
                'classes_per_client': self.classes_per_client,
                'examples_per_client': self.examples_per_client}
