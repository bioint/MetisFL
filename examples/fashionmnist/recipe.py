
import numpy as np

from metisfl.models.model_dataset import ModelDataset, ModelDatasetClassification


def dataset_recipe_fn(dataset_fp: str) -> ModelDataset:
    """A dataset recipe function that loads a dataset from a file path.

    Args:
        dataset_fp (str): The file path to the dataset.

    Returns:
        ModelDataset: _description_
    """
    
    # Load the dataset
    loaded_dataset = np.load(dataset_fp)
    
    # Get the x and y values
    x, y = loaded_dataset['x'], loaded_dataset['y']
    
    # Get the classes and their counts
    unique, counts = np.unique(y, return_counts=True)
    
    # Create a dictionary with the examples per class
    examples_per_class = {}
    for class_id, class_counts in zip(unique, counts):
        examples_per_class[class_id] = class_counts
        
    # Create a ModelDataset object, in this case a ModelDatasetClassification
    model_dataset = ModelDatasetClassification(
        x=x, y=y, size=y.size, examples_per_class=examples_per_class)
    
    # Return the ModelDataset object
    return model_dataset
