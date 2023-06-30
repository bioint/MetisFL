import argparse
import json
import os

import numpy as np
import tensorflow as tf

from examples.keras.models.fashion_mnist_fc import get_model
from examples.utils.data_partitioning import DataPartitioning
from metisfl.driver.driver_session import DriverSession
from metisfl.models.keras.wrapper import MetisKerasModel
from metisfl.models.model_dataset import ModelDatasetClassification
from metisfl.utils.fedenv_parser import FederationEnvironment

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    default_federation_environment_config_fp = os.path.join(
        script_cwd, "../config/cifar10/test_localhost_synchronous_vanillasgd.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--federation_environment_config_fp",
                        default=default_federation_environment_config_fp)
    parser.add_argument("--generate_iid_partitions", default=False)
    parser.add_argument("--generate_noniid_partitions", default=False)

    args = parser.parse_args()
    print(args, flush=True)

    """ Load the environment. """
    federation_environment = FederationEnvironment(args.federation_environment_config_fp)

    """ Load the data. """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(x_train[0:1].shape, flush=True)
    print(y_train[0:1].shape, flush=True)
    x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
    x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

    if not args.generate_iid_partitions and not args.generate_noniid_partitions \
            and not all([l.dataset_configs.train_dataset_path for l in federation_environment.learners]):
        raise RuntimeError("Need to specify datasets training paths or pass generate iid/noniid partitions argument.")

    if args.generate_iid_partitions or args.generate_noniid_partitions:
        # Parse environment to assign datasets to learners.
        num_learners = len(federation_environment.learners.learners)
        if args.generate_iid_partitions:
            x_chunks, y_chunks = DataPartitioning(x_train, y_train, num_learners).iid_partition()
        if args.generate_noniid_partitions:
            num_learners = len(federation_environment.learners.learners)
            x_chunks, y_chunks = DataPartitioning(x_train, y_train, num_learners) \
                .non_iid_partition(classes_per_partition=2)

        datasets_path = os.path.join(script_cwd, "datasets/fashionmnist/")
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)
        np.savez(os.path.join(datasets_path, "test.npz"), x=x_test, y=y_test)
        for cidx, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
            np.savez(os.path.join(datasets_path, "train_{}.npz".format(cidx)), x=x_chunk, y=y_chunk)
        for lidx, learner in enumerate(federation_environment.learners.learners):
            learner.dataset_configs.test_dataset_path = \
                os.path.join(datasets_path, "test.npz")
            learner.dataset_configs.train_dataset_path = \
                os.path.join(datasets_path, "train_{}.npz".format(lidx))

    nn_model = get_model()
    # Perform an .evaluation() step to initialize all Keras 'hidden' states, else model.save() will not save the model
    # properly and any subsequent fit step will never train the model properly. We could apply the .fit() step instead
    # of the .evaluation() step, but since the driver does not hold any data it simply evaluates a random sample.
    nn_model.evaluate(x=np.random.random(x_train[0:1].shape), y=np.random.random(y_train[0:1].shape), verbose=False)
    nn_model = MetisKerasModel(nn_model)

    def dataset_recipe_fn(dataset_fp):
        loaded_dataset = np.load(dataset_fp)
        x, y = loaded_dataset['x'], loaded_dataset['y']
        unique, counts = np.unique(y, return_counts=True)
        distribution = {}
        for cid, num in zip(unique, counts):
            distribution[cid] = num
        model_dataset = ModelDatasetClassification(
            x=x, y=y, size=y.size, examples_per_class=distribution)
        return model_dataset

    driver_session = DriverSession(federation_environment,
                                   nn_model,
                                   train_dataset_recipe_fn=dataset_recipe_fn,
                                   test_dataset_recipe_fn=dataset_recipe_fn)
    driver_session.initialize_federation()
    driver_session.monitor_federation()
    driver_session.shutdown_federation()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
