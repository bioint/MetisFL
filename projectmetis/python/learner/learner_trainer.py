from projectmetis.python.models.model_dataset import ModelDataset


class LearnerTrainer(object):

    def __init__(self, model_ops_fn):
        self._model_ops = model_ops_fn()

    def train_model(self, dataset: ModelDataset, learning_task_pb, hyperparameters_pb,
                    model_pb, verbose=False, *args, **kwargs):
        num_updates = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        variables_pb = model_pb.variables
        var_names, var_trainables, var_nps = \
            self._model_ops.get_model_weights_from_variables_pb(variables_pb)
        optimizer = self._model_ops.construct_optimizer(
            optimizer_config_pb=hyperparameters_pb.optimizer)
        # TODO Compile model with new optimizer
        # Assign new model weights after model compilation.
        if len(var_nps) > 0:
            self._model_ops.set_model_weights(var_nps)
        training_res = self._model_ops.train_model(dataset, num_updates, batch_size, verbose)
        trained_model = self._model_ops.get_model_weights()
        return training_res, trained_model
