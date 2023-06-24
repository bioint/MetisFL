
import cloudpickle
from inspect import signature
from pebble import ProcessPool

from metisfl.models.model_dataset import ModelDataset


class LearnerDataset:
    def __init__(self,
                 train_dataset_fp, 
                 train_dataset_recipe_pkl,
                 validation_dataset_fp="", 
                 validation_dataset_recipe_pkl="",
                 test_dataset_fp="", 
                 test_dataset_recipe_pkl=""):
        if not train_dataset_recipe_pkl:
            raise AssertionError("Training dataset recipe is required.")
 
        self.train_dataset_recipe_pkl, self.train_dataset_fp = \
            train_dataset_recipe_pkl, train_dataset_fp
        self.validation_dataset_recipe_pkl, self.validation_dataset_fp = \
            validation_dataset_recipe_pkl, validation_dataset_fp
        self.test_dataset_recipe_pkl, self.test_dataset_fp = \
            test_dataset_recipe_pkl, test_dataset_fp

    def load_model_datasets(self):
        train_dataset = create_model_dataset_helper(
            self.train_dataset_recipe_pkl, self.train_dataset_fp)
        validation_dataset = create_model_dataset_helper(
            self.validation_dataset_recipe_pkl, self.validation_dataset_fp,
            default_class=train_dataset.__class__)
        test_dataset = create_model_dataset_helper(
            self.test_dataset_recipe_pkl, self.test_dataset_fp,
            default_class=train_dataset.__class__)
        return train_dataset, validation_dataset, test_dataset

    def load_model_datasets_size_specs_type_def(self):
        # Load only the dataset size, specifications and class type because
        # numpys or tf.tensors cannot be serialized and hence cannot be returned through the process.
        return [(d.get_size(), d.get_model_dataset_specifications(), type(d)) for d in self.load_model_datasets()]

    # @stripeli why do we need to load the dataset in a subprocess?
    def load_datasets_metadata_subproc(self):
        _generic_tasks_pool = ProcessPool(max_workers=1, max_tasks=1)
        datasets_specs_future = _generic_tasks_pool.schedule(function=self.load_model_datasets_size_specs_type_def)
        res = datasets_specs_future.result()
        _generic_tasks_pool.close()
        _generic_tasks_pool.join()
        return res

def create_model_dataset_helper(dataset_recipe_pkl, dataset_fp=None, default_class=None):
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
