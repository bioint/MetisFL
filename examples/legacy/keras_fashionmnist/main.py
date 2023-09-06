import argparse
import json
import os

from dataset import load_data, partition_data_iid, save_data
from model import get_model
from recipe import dataset_recipe_fn

from metisfl.driver.driver import DriverSession
from metisfl.models.keras.keras_model import MetisModelKeras
from metisfl.models.keras.optimizers.fed_prox import FedProx

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    script_cwd = os.path.abspath(script_cwd)
    default_fed_env_fp = os.path.join(
        script_cwd, "../fedenv_templates/template_new.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=default_fed_env_fp)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--num_learners", default=1)
    args = parser.parse_args()

    # Load the data
    x_train, y_train, x_test, y_test = load_data()

    # Partition the data, iid partitions
    # Must match the number of learners in the federation environment yaml file
    num_learners = int(args.num_learners)
    x_chunks, y_chunks = partition_data_iid(x_train, y_train, num_learners)

    # Save the data
    train_dataset_fps, test_dataset_fp = save_data(
        x_chunks, y_chunks, x_test, y_test)

    # Replicate the test dataset for each learner; all learners use the same test dataset
    test_dataset_fps = [test_dataset_fp for _ in range(num_learners)]

    # Get the tf.keras model
    model = get_model()
    optimizer = args.optimizer
    if optimizer == "FedProx":
        optimizer = FedProx()
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    # Wrap the model in a MetisKerasModel
    metis_model = MetisModelKeras(model)

    # Create a DriverSession
    driver_session = DriverSession(fed_env=args.env,
                                   model=metis_model,
                                   train_dataset_recipe_fn=dataset_recipe_fn,
                                   train_dataset_fps=train_dataset_fps,
                                   validation_dataset_fps=test_dataset_fps,
                                   validation_dataset_recipe_fn=dataset_recipe_fn,
                                   test_dataset_recipe_fn=dataset_recipe_fn,
                                   test_dataset_fps=test_dataset_fps)

    # Run the DriverSession
    driver_session.run()

    # Get the statistics
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
