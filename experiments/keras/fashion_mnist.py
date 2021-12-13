import cloudpickle
import os

import numpy as np
import tensorflow as tf

from experiments.keras.models.fashion_mnist_fc import FashionMnistModel
from projectmetis.python.driver.driver_session import DriverSession
from projectmetis.python.models.model_dataset import ModelDatasetClassification


if __name__ == "__main__":

    nn_engine = "keras"
    metis_filepath_prefix = "/tmp/metis/model/"
    if not os.path.exists(metis_filepath_prefix):
        os.makedirs(metis_filepath_prefix)

    model_filepath = "{}/model_definition".format(metis_filepath_prefix)
    train_dataset_filepath = "{}/model_train_dataset.npz".format(metis_filepath_prefix)
    validation_dataset_filepath = "{}/model_validation_dataset.npz".format(metis_filepath_prefix)
    test_dataset_filepath = "{}/model_test_dataset.npz".format(metis_filepath_prefix)
    train_dataset_recipe_fp_pkl = "{}/model_train_dataset_ops.pkl".format(metis_filepath_prefix)
    validation_dataset_recipe_fp_pkl = "{}/model_validation_dataset_ops.pkl".format(metis_filepath_prefix)
    test_dataset_recipe_fp_pkl = "{}/model_test_dataset_ops.pkl".format(metis_filepath_prefix)

    fmnist_model = FashionMnistModel().get_model()
    # TODO Save model as tf native and ship the resulted files.
    """ Load the data. """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_valid, y_valid = x_train[:6000], y_train[:6000]
    x_train, y_train = x_train[6000:], y_train[6000:]
    print(x_train[0:1].shape)
    print(y_train[0:1].shape)
    x_train = (x_train.astype('float32') / 256).reshape(-1, 28, 28, 1)
    x_valid = (x_valid.astype('float32') / 256).reshape(-1, 28, 28, 1)
    x_test = (x_test.astype('float32') / 256).reshape(-1, 28, 28, 1)

    # Save data.
    np.savez(train_dataset_filepath, x=x_train, y=y_train)
    np.savez(validation_dataset_filepath, x=x_valid, y=y_valid)
    np.savez(test_dataset_filepath, x=x_test, y=y_test)

    nn_model = fmnist_model
    # Perform an .evaluation() step to initialize all Keras 'hidden' states, else model.save() will not save the model
    # properly and any subsequent fit step will never train the model properly. We could apply the .fit() step instead
    # of the .evaluation() step, but since the driver does not hold any data it simply evaluates a random sample.
    nn_model.evaluate(x=np.random.random(x_train[0:1].shape), y=np.random.random(y_train[0:1].shape), verbose=False)
    nn_model.save(model_filepath)

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

    cloudpickle.dump(obj=dataset_recipe_fn, file=open(train_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(test_dataset_recipe_fp_pkl, "wb+"))
    cloudpickle.dump(obj=dataset_recipe_fn, file=open(validation_dataset_recipe_fp_pkl, "wb+"))

    # dirname = os.path.dirname(__file__)
    # federation_environment_config_fp = os.path.join(
    #     dirname, "../federation_environments_config/test_localhost.yaml")
    # driver_session = DriverSession(federation_environment_config_fp, nn_engine,
    #                                model_filepath=model_filepath,
    #                                train_dataset_recipe_fp=train_dataset_recipe_fp_pkl,
    #                                validation_dataset_recipe_fp=validation_dataset_recipe_fp_pkl,
    #                                test_dataset_recipe_fp=test_dataset_recipe_fp_pkl)
    # driver_session.initialize_federation()
    # # when
    # driver_session.monitor_federation()
    # driver_session.shutdown_federation()
