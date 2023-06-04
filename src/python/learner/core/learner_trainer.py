from src.python.models.model_dataset import ModelDataset


class LearnerTrainer(object):

    def __init__(self, model_ops_fn, model_pb):
        """
        The initial state of the model for every other trainer instance is unique. Therefore,
        we initialize the model once during the trainer object construction time.
        :param model_ops_fn:
        :param model_pb:
        """
        self._model_ops = model_ops_fn()
        self.model_pb = model_pb
        self.weights_names, self.weights_trainable, self.weights_values = \
            self._model_ops.get_model_weights_from_variables_pb(self.model_pb.variables)
        if len(self.weights_values) > 0:
            self._model_ops.set_model_weights(self.weights_names, self.weights_trainable, self.weights_values)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model_ops.cleanup()

    def train_model(self, train_dataset: ModelDataset, learning_task_pb, hyperparameters_pb,
                    validation_dataset=None, test_dataset=None, verbose=False, *args, **kwargs):
        completed_task_pb = self._model_ops\
            .train_model(train_dataset, learning_task_pb, hyperparameters_pb,
                         validation_dataset, test_dataset, verbose)
        return completed_task_pb
