import metisfl.utils.proto_messages_factory as proto_factory

from metisfl.encryption import fhe
from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.learner.weight_decrypt import get_model_weights_from_variables_pb
from metisfl.proto import learner_pb2, model_pb2, metis_pb2
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.metis_logger import MetisLogger


class LearnerEvaluator(object):

    def __init__(self, 
                 learner_dataset: LearnerDataset,
                 nn_engine, 
                 model_dir, 
                 he_scheme):
        """
        We apply the model weights one-time during the initialization of the Evaluator constructor
        because applying the weights on per evaluation or inference operation for large models,
        it delays dramatically the execution time of the whole pipeline.
        """
        self.learner_dataset = learner_dataset
        self.model_dir = model_dir
        self.he_scheme = he_scheme
        self._setup_fhe()
        self._model_ops = self._get_model_ops(nn_engine) # nn_engine = "keras" or "pytorch"
        
    def _setup_fhe(self):
        he_scheme = None
        if self.he_scheme.enabled:
            if self.he_scheme.HasField("fhe_scheme"):
                he_scheme = fhe.CKKS(
                    self.he_scheme.fhe_scheme.batch_size,
                    self.he_scheme.fhe_scheme.scaling_bits,
                    "resources/fheparams/cryptoparams/")
                he_scheme.load_crypto_params()
                
    def _get_model_ops(self, nn_engine):
        if nn_engine == "keras":
            from metisfl.models.keras.keras_model_ops import KerasModelOps
            return KerasModelOps(model_dir=self.model_dir)
        if nn_engine == "pytorch":
            from metisfl.models.pytorch.pytorch_model_ops import PyTorchModelOps
            return PyTorchModelOps(model_dir=self.model_dir)

    def _set_weights_from_model_pb(self, model_pb):
        weights_names, weights_trainable, weights_values = \
            get_model_weights_from_variables_pb(model_pb.variables, self.he_scheme)
        if len(self.weights_values) > 0:
            self._model_ops.get_model().set_model_weights(self.weights_names, self.weights_trainable, self.weights_values)
        return weights_names, weights_trainable, weights_values

    def evaluate_model(self, 
                        model_pb: model_pb2.Model, 
                        batch_size: int,
                        evaluation_datasets_pb: [learner_pb2.EvaluateModelRequest.dataset_to_eval],
                        metrics_pb: metis_pb2.EvaluationMetrics, 
                        verbose=False):
        MetisLogger.info("Learner {} starts model evaluation on requested datasets."
                         .format(self.host_port_identifier()))        
        
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
            
        # Need to unfold the pb into python list.
        metrics = [m for m in metrics_pb.metric]
                
        # Initialize to an empty metis_pb2.ModelEvaluation object all three variables.
        train_eval_pb = validation_eval_pb = test_eval_pb = metis_pb2.ModelEvaluation()
        for dataset_to_eval in evaluation_datasets_pb:
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TRAINING:
                train_eval_pb = self._model_ops.evaluate_model(train_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.VALIDATION:
                validation_eval_pb = self._model_ops.evaluate_model(validation_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TEST:
                test_eval_pb = self._model_ops.evaluate_model(test_dataset, batch_size, metrics, verbose)
                                
        model_evaluations_pb = \
            proto_factory.MetisProtoMessages.construct_model_evaluations_pb(
                training_evaluation_pb=train_eval_pb,
                validation_evaluation_pb=validation_eval_pb,
                test_evaluation_pb=test_eval_pb)
        MetisLogger.info("Learner {} completed model evaluation on requested datasets."
                            .format(self.host_port_identifier()))

        return model_evaluations_pb

    def infer_model(self, 
                    model_pb: model_pb2.Model, 
                    batch_size: int,
                    infer_train=False, 
                    infer_test=False, 
                    infer_valid=False, 
                    verbose=False):
        MetisLogger.info("Learner {} starts model inference on requested datasets."
                         .format(self.host_port_identifier()))
        # TODO infer model should behave similarly as the evaluate_model(), by looping over a
        #  similar learner_pb2.InferModelRequest.dataset_to_infer list.
        
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
        inferred_res = {
            "train": self._model_ops.infer_model(train_dataset, batch_size, verbose) if infer_train else None, 
            "valid": self._model_ops.infer_model(validation_dataset, batch_size, verbose) if infer_valid else None,
            "test": self._model_ops.infer_model(test_dataset, batch_size, verbose) if infer_test else None
        }            
        stringified_res = DictionaryFormatter.stringify(inferred_res, stringify_nan=True)
        
        MetisLogger.info("Learner {} completed model inference on requested datasets."
                            .format(self.host_port_identifier()))
        return stringified_res
        
    def train_model(self, 
                    model_pb: model_pb2.Model,
                    learning_task_pb, 
                    hyperparameters_pb,
                    verbose=False):
        MetisLogger.info("Learner {} starts model training on local training dataset."
                    .format(self.host_port_identifier()))

        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()            
        completed_task_pb = self._model_ops\
            .train_model(train_dataset, learning_task_pb, hyperparameters_pb,
                         validation_dataset, test_dataset, verbose)
        
        MetisLogger.info("Learner {} completed model training on local training dataset."
                            .format(self.host_port_identifier()))
        return completed_task_pb

    def __enter__(self):
        return self

    def __exit__(self):
        self._model_ops.cleanup()

