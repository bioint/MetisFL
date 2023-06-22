import cloudpickle
from inspect import signature

from metisfl.models.model_dataset import ModelDataset

def create_model_dataset_helper(dataset_recipe_pkl, dataset_fp=None, default_class=None):
    # TODO Move into utils? @stripeli YES :) This is a static function. No "self" is used.
    """
    Thus function loads the dataset recipe dynamically. To achieve this, we
    need to see if the given recipe takes any arguments. The only argument
    we expect to be given is the path to the dataset (filepath).
    Therefore, we need to check the function's arguments
    cardinality if it is greater than 0.
    :param dataset_recipe_pkl:
    :param dataset_fp:
    :param default_class:
    :return:
    """

    if not dataset_recipe_pkl and not default_class:
        raise RuntimeError("Neither the dataset recipe or the default class are specified. Exiting ...")

    if dataset_recipe_pkl:
        dataset_recipe_fn = cloudpickle.load(open(dataset_recipe_pkl, "rb"))
        fn_params = signature(dataset_recipe_fn).parameters.keys()
        if len(fn_params) > 0:
            if dataset_fp:
                # If the function expects an input we pass the dataset path.
                dataset = dataset_recipe_fn(dataset_fp)
            else:
                # If the dataset recipe requires an input file but none was given
                # then we will return the default class.
                dataset = default_class()
        else:
            # Else we just load the dataset as is.
            # This represents the in-memory dataset loading.
            dataset = dataset_recipe_fn()
    else:
        dataset = default_class()

    assert isinstance(dataset, ModelDataset), \
        "The dataset needs to be an instance of: {}".format(ModelDataset.__name__)
    return dataset
