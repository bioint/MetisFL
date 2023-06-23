
from metisfl.learner.utils import create_model_dataset_helper
from pebble import ProcessPool

class LearnerDataset:
    def __init__(self,
                 train_dataset_fp, 
                 train_dataset_recipe_pkl,
                 validation_dataset_fp="", 
                 validation_dataset_recipe_pkl="",
                 test_dataset_fp="", 
                 test_dataset_recipe_pkl=""
        ):
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
