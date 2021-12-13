import cloudpickle
import os

import numpy as np
import tensorflow as tf

from experiments.keras.models.cifar_cnn import CifarCNN
from projectmetis.proto.metis_pb2 import ServerEntity, DatasetSpec
from projectmetis.python.learner.learner import Learner
from projectmetis.python.learner.learner_servicer import LearnerServicer
from projectmetis.python.models.model_dataset import ModelDataset


if __name__ == "__main__":
    # ARGS
    # 1. Learner hostname
    # 2. Learner port: "Docker" port that the learner servicer will be listening on
    # 3. Controller hostname
    # 4. Controller port
    # 5. Model definition/architecture/structure
    #       Method 1: Send .gz file from the driver and store inside /tmp/
    #       Method 2: Create an endpoint to the LearnerServicer and send it there after servicer is initialized
    # 6. Path to training dataset
    # 7. Path to validation dataset
    # 8. Path to test dataset
    # 9. Datasets recipe: one for each dataset!

    metis_filepath_prefix = "/tmp/projectmetis"
    if not os.path.exists(metis_filepath_prefix):
        os.makedirs(metis_filepath_prefix)

    model_filepath = "{}/model_definition".format(metis_filepath_prefix)
    train_dataset_filepath = "{}/model_train_dataset.npz".format(metis_filepath_prefix)
    valid_dataset_filepath = "{}/model_valid_dataset.npz".format(metis_filepath_prefix)
    test_dataset_filepath = "{}/model_test_dataset.npz".format(metis_filepath_prefix)
    dataset_recipe_fp_pkl = "{}/model_dataset_ops.pkl".format(metis_filepath_prefix)

    cifar10_model = CifarCNN().get_model()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_valid, y_valid = x_train[:6000], y_train[:6000]
    x_train, y_train = x_train[6000:], y_train[6000:]
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Save data.
    np.savez(train_dataset_filepath, x=x_train, y=y_train)
    np.savez(valid_dataset_filepath, x=x_valid, y=y_valid)
    np.savez(test_dataset_filepath, x=x_test, y=y_test)

    nn_model = cifar10_model
    # Perform an .evaluation() step to initialize all Keras 'hidden' states, else model.save() will not save the model
    # properly and any subsequent fit step will never train the model properly. We could apply the .fit() step instead
    # of the .evaluation() step, but since the driver does not hold any data it simply evaluates a random sample.
    nn_model.evaluate(x=np.random.random(x_train[0:1].shape), y=np.random.random(y_train[0:1].shape), verbose=False)
    nn_model.save(model_filepath)

    # TODO Check serialization of the recipe through cloudpickle -
    #  serialized recipe pass as arg
    def dataset_recipe_fn(dataset_fp):
        loaded_dataset = np.load(dataset_fp)
        x, y = loaded_dataset['x'], loaded_dataset['y']
        train_dataset = ModelDataset(x=x, y=y, size=y.size)
        return train_dataset
    cloudpickle.dump(dataset_recipe_fn, open(dataset_recipe_fp_pkl, "wb+"))
    dataset_recipe_fn = cloudpickle.load(open(dataset_recipe_fp_pkl, "rb"))

    # TODO Combine (train_dataset_recipe_fn and train_dataset_fp) and define
    #  a load() function that invokes all the data and the defined ETLs.
    learner_id = "TestLearner"
    learner_server_entity = ServerEntity(hostname="[::]", port=50052)
    controller_server_entity = ServerEntity(hostname="0.0.0.0", port=50051)
    learner = Learner(
        learner_server_entity=learner_server_entity,
        controller_server_entity=controller_server_entity,
        nn_engine="keras",
        model_fp=model_filepath,
        train_dataset_fp=train_dataset_filepath,
        train_dataset_recipe_fn=dataset_recipe_fn,
        test_dataset_fp=test_dataset_filepath,
        test_dataset_recipe_fn=dataset_recipe_fn,
        validation_dataset_fp=valid_dataset_filepath,
        validation_dataset_recipe_fn=dataset_recipe_fn)
    learner.join_federation()
    learner_servicer = LearnerServicer(
        learner=learner,
        learner_server_entity=learner_server_entity,
        servicer_workers=10)
    learner_servicer.init_servicer()
    learner_servicer.wait_servicer()
    learner.leave_federation()
