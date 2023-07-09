import argparse
import json
import os

from dataset import load_data, partition_data_iid, save_data
from model import get_model
from recipe import dataset_recipe_fn

from metisfl.driver.driver_session import DriverSession
from metisfl.models.keras.wrapper import MetisKerasModel
from metisfl.utils.fedenv_parser import FederationEnvironment

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    script_cwd = os.path.abspath(script_cwd)
    default_fed_env_fp = os.path.join(script_cwd, "./envs/test_localhost_synchronous_vanillasgd.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=default_fed_env_fp)
    args = parser.parse_args()

    # Load the environment configuration
    federation_environment = FederationEnvironment(args.env)

    # Load the data
    x_train, y_train, x_test, y_test = load_data()
    
    # Partition the data, iid partitions
    num_learners = len(federation_environment.learners)
    x_chunks, y_chunks = partition_data_iid(x_train, y_train, num_learners)
    
    # Save the data
    train_dataset_fps, test_dataset_fp = save_data(x_chunks, y_chunks, x_test, y_test)

    # Replicate the test dataset for each learner; all learners use the same test dataset
    test_dataset_fps = [test_dataset_fp for _ in range(num_learners)]

    # Get params from the environment yaml file
    optimizer_config = federation_environment.local_model_config.optimizer_config
    metric = federation_environment.evaluation_metric
    
    # Get the model
    nn_model = get_model(metrics=[metric], optimizer_config=optimizer_config)

    # Wrap it with MetisKerasModel
    model = MetisKerasModel(nn_model)

    # Create the driver session
    driver_session = DriverSession(fed_env=federation_environment,
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
