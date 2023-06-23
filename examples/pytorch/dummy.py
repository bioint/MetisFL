import argparse
import json
import os

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from examples.pytorch.models.mlp import MLP
from metisfl.driver.driver_session import DriverSession
from metisfl.models.model_dataset import ModelDatasetClassification
from metisfl.models.pytorch.wrapper import MetisTorchModel
from metisfl.utils.fedenv_parser import FederationEnvironment


# Dummy dataset definition.
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def train_test_split(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    print("Script current working directory: ", script_cwd, flush=True)
    default_federation_environment_config_fp = os.path.join(
        script_cwd, "../config/template_with_ssl.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--federation_environment_config_fp",
                        default=default_federation_environment_config_fp)

    args = parser.parse_args()
    print(args, flush=True)

    """ Load the environment. """
    federation_environment = FederationEnvironment(args.federation_environment_config_fp)

    # # Prepare the data.
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'

    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.train_test_split(n_test=0.1)

    def train_dataset_recipe_fn():
        train_dl = DataLoader(train, batch_size=32, shuffle=True)
        model_dataset = ModelDatasetClassification(
            x=train_dl, size=len(train_dl.dataset))
        return model_dataset

    def test_dataset_recipe_fn():
        test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        model_dataset = ModelDatasetClassification(
            x=test_dl, size=len(test_dl.dataset))
        return model_dataset

    # Define the network.
    model = MLP(n_inputs=34)

    driver_session = DriverSession(federation_environment,
                                   model=MetisTorchModel(model),
                                   train_dataset_recipe_fn=train_dataset_recipe_fn,
                                   validation_dataset_recipe_fn=None,
                                   test_dataset_recipe_fn=None)
    driver_session.initialize_federation()
    driver_session.monitor_federation()
    driver_session.shutdown_federation()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
