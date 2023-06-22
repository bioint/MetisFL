from metisfl.models.model_dataset import ModelDataset
from metisfl.encryption import fhe
from metisfl.learner.weight_decrypt import get_model_weights_from_variables_pb


class LearnerEvaluator(object):

    def __init__(self, nn_engine, model_dir, he_scheme, model_pb):
        """
        We apply the model weights one-time during the initialization of the Evaluator constructor
        because applying the weights on per evaluation or inference operation for large models,
        it delays dramatically the execution time of the whole pipeline.
        """
        self._nn_engine = nn_engine
        self._model_dir = model_dir
        self._he_scheme = he_scheme
        self._model_pb = model_pb
        
        self._setup_fhe()
        
        self.weights_names, self.weights_trainable, self.weights_values = \
            get_model_weights_from_variables_pb(self._model_pb.variables, self._he_scheme)
            
        self._model_ops = self._get_model_ops(nn_engine)()
        if len(self.weights_values) > 0:
            self._model_ops.get_model().set_model_weights(self.weights_names, self.weights_trainable, self.weights_values)

    def _setup_fhe(self):
        he_scheme = None
        if self._he_scheme.enabled:
            if self._he_scheme.HasField("fhe_scheme"):
                he_scheme = fhe.CKKS(
                    self._he_scheme.fhe_scheme.batch_size,
                    self._he_scheme.fhe_scheme.scaling_bits,
                    "resources/fheparams/cryptoparams/")
                he_scheme.load_crypto_params()

    def evaluate_model(self, dataset: ModelDataset, 
                       batch_size,
                       metrics=None, 
                       verbose=False):
        model_eval_pb = self._model_ops.evaluate_model(dataset, batch_size, metrics, verbose)
        return model_eval_pb

    def infer_model(self, dataset: ModelDataset,
                    batch_size,
                    verbose=False):
        infer_res = self._model_ops.infer_model(dataset, batch_size, verbose)
        return infer_res
    
    def train_model(self, 
                    train_dataset: ModelDataset, 
                    learning_task_pb, 
                    hyperparameters_pb,
                    validation_dataset=None, 
                    test_dataset=None, 
                    verbose=False):
        completed_task_pb = self._model_ops\
            .train_model(train_dataset, learning_task_pb, hyperparameters_pb,
                         validation_dataset, test_dataset, verbose)
        return completed_task_pb


    def _get_model_ops(self, nn_engine):
        if nn_engine == "keras":
            from metisfl.models.keras.keras_model_ops import KerasModelOps
            return KerasModelOps(model_dir=self._model_dir)
        if nn_engine == "pytorch":
            from metisfl.models.pytorch.pytorch_model_ops import PyTorchModelOps
            return PyTorchModelOps(model_dir=self._model_dir)

    def __enter__(self):
        return self

    def __exit__(self):
        self._model_ops.cleanup()

