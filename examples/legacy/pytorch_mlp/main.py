import argparse
import json
import os

from dataset import CSVDataset
from mlp import MLP
from torch.utils.data import DataLoader

from metisfl.driver.driver import DriverSession
from metisfl.models.model_dataset import ModelDatasetClassification
from metisfl.models.torch.torch_learner import MetisModelTorch

if __name__ == "__main__":

    script_cwd = os.path.dirname(__file__)
    env_config = os.path.join(script_cwd, "template.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=env_config)
    args = parser.parse_args()

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

    # Wrap the model.
    model = MetisModelTorch(model)

    driver_session = DriverSession(args.env,
                                   model=model,
                                   train_dataset_fps=[path],
                                   train_dataset_recipe_fn=train_dataset_recipe_fn)

    driver_session.run()
    statistics = driver_session.get_federation_statistics()

    with open(os.path.join(script_cwd, "experiment.json"), "w+") as fout:
        print("Execution File Output Path:", fout.name, flush=True)
        json.dump(statistics, fout, indent=4)
