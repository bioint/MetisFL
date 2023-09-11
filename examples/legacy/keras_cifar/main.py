import argparse
import json
import os

from dataset import load_data, partition_data_iid, save_data
from model import get_model
from recipe import dataset_recipe_fn

from metisfl.driver.driver import DriverSession
from metisfl.models.keras.keras_model import MetisModelKeras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    script_cwd = os.path.abspath(script_cwd)
    default_fed_env_fp = os.path.join(script_cwd, "template.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=default_fed_env_fp)
    args = parser.parse_args()

    # Load the data
    x_train, y_train, x_test, y_test = load_data()

    # Partition the data, iid partitions
    num_learners = 1  # must match the number of learners in the environment yaml file
    x_chunks, y_chunks = partition_data_iid(x_train, y_train, num_learners)

    # Save the data
    train_dataset_fps, test_dataset_fp = save_data(
        x_chunks, y_chunks, x_test, y_test)

    # Replicate the test dataset for each learner; all learners use the same test dataset
    test_dataset_fps = [test_dataset_fp for _ in range(num_learners)]

    # Get the model
    nn_model = get_model()

    # Compile the model
    nn_model.compile(optimizer="adam",
                     loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Wrap it with MetisKerasModel
    model = MetisModelKeras(nn_model)

    # Create the driver session
    driver_session = DriverSession(fed_env=args.env,
                                   model=model,
                                   train_dataset_fps=train_dataset_fps,
                                   test_dataset_fps=test_dataset_fps,
                                   validation_dataset_fps=test_dataset_fps,
                                   validation_dataset_recipe_fn=dataset_recipe_fn,
                                   train_dataset_recipe_fn=dataset_recipe_fn,
                                   test_dataset_recipe_fn=dataset_recipe_fn)

    # Run the driver session
    driver_session.run()

    # Get the statistics
    statistics = driver_session.get_federation_statistics()

    # Save the statistics
    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
