import argparse
import json
import os

from metisfl.driver.driver_session import DriverSession
from metisfl.models.keras.wrapper import MetisKerasModel
from metisfl.utils.fedenv_parser import FederationEnvironment

from .dataset import load_data, partition_data_iid, save_data
from .model import get_model
from .receipe import dataset_recipe_fn


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    default_fed_env_fp = os.path.join("envs/test_localhost_synchronous_vanillasgd.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--federation_environment_config_fp", default=default_fed_env_fp)
    parser.add_argument("--generate_iid_partitions", default=False)
    parser.add_argument("--generate_noniid_partitions", default=False)
    args = parser.parse_args()

    # Load the environment configuration
    federation_environment = FederationEnvironment(args.federation_environment_config_fp)

    # Load the data
    x_train, y_train, x_test, y_test = load_data()
    
    # Partition the data, iid partitions
    x_chunks, y_chunks = partition_data_iid(x_train, y_train, federation_environment.num_learners)
    
    # Save the data
    train_dataset_fps, test_dataset_fp = save_data(x_chunks, y_chunks, x_test, y_test)

    # Get the tf.keras model
    model = get_model()
    
    # Wrap the model in a MetisKerasModel
    metis_model = MetisKerasModel(model)

    # Create a DriverSession
    driver_session = DriverSession(federation_environment,
                                    model=metis_model,
                                    train_dataset_recipe_fn=dataset_recipe_fn,
                                    test_dataset_recipe_fn=dataset_recipe_fn)
    
    driver_session.initialize_federation()
    driver_session.monitor_federation()
    driver_session.shutdown_federation()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
