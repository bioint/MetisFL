from projectmetis.python.models.model_dataset import ModelDataset


class LearnerEvaluator(object):

    def __init__(self, model_ops_fn):
        self._model_ops = model_ops_fn()

    def evaluate_model(self, dataset: ModelDataset, model_pb, batch_size,
                       metrics=None, verbose=False, *args, **kwargs):
        variables_pb = model_pb.variables
        var_names, var_trainables, var_nps = \
            self._model_ops.get_model_weights_from_variables_pb(variables_pb)
        if len(var_nps) > 0:
            self._model_ops.set_model_weights(var_nps)
        eval_res = self._model_ops.evaluate_model(dataset, batch_size, metrics, verbose)
        return eval_res

    def infer_model(self, dataset: ModelDataset, model_pb, batch_size,
                    verbose=False, *args, **kwargs):
        variables_pb = model_pb.variables
        var_names, var_trainables, var_nps = \
            self._model_ops.get_model_weights_from_variables_pb(variables_pb)
        if len(var_nps) > 0:
            self._model_ops.set_model_weights(var_nps)
        infer_res = self._model_ops.infer_model(dataset, batch_size, verbose)
        return infer_res
