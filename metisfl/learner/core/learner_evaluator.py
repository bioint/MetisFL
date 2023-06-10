from metisfl.learner.models.model_dataset import ModelDataset


class LearnerEvaluator(object):

    def __init__(self, model_ops_fn, model_pb):
        """
        We apply the model weights one-time during the initialization of the Evaluator constructor
        because applying the weights on per evaluation or inference operation for large models,
        it delays dramatically the execution time of the whole pipeline.
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

    def evaluate_model(self, dataset: ModelDataset, batch_size,
                       metrics=None, verbose=False, *args, **kwargs):
        model_eval_pb = self._model_ops.evaluate_model(dataset, batch_size, metrics, verbose)
        return model_eval_pb

    def infer_model(self, dataset: ModelDataset, batch_size,
                    verbose=False, *args, **kwargs):
        infer_res = self._model_ops.infer_model(dataset, batch_size, verbose)
        return infer_res
