from typing import Callable

from dataset_handler import LearnerDataset

from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.models import ModelOps
from metisfl.models.utils import get_completed_learning_task_pb
from metisfl.proto import learner_pb2, metis_pb2, model_pb2
from metisfl.utils import DictionaryFormatter
from metisfl.utils.metis_logger import MetisLogger


class TaskExecutor(object):

    def __init__(self, 
                 he_scheme_pb: metis_pb2.HEScheme,
                 learner_dataset: LearnerDataset,
                 model_dir: str,
                 model_ops_fn: Callable[[str], ModelOps]):
        """A class that executes training/evaluation/inference tasks. The tasks in this class are
            executed in a independent process, different from the process that created the object. 
            It is importart to call the init_model_backend() method before calling any other method. 
            And it has to be called within the same process that runs the tasks so that the model
            backend is imported correctly.    

        Args:
            he_scheme_pb (metis_pb2.HEScheme): A protobuf message that contains the HE scheme.
            learner_dataset (LearnerDataset): A LearnerDataset object that contains the datasets.
            model_backend_fn (Callable[[str], model_ops.ModelOps]): A function that returns a model backend.
            model_dir (str): The directory where the model is stored.
        """
        self.homomorphic_encryption = HomomorphicEncryption.from_proto(he_scheme_pb)
        self.learner_dataset = learner_dataset
        self.model_ops = None 
        self.model_ops_fn = model_ops_fn
        self.model_dir = model_dir
        
    def _init_model_ops(self) -> ModelOps:
        if not self.model_ops:
            self.model_ops = self.model_ops_fn(self.model_dir)
        
    def _set_weights_from_model_pb(self, model_pb: model_pb2.Model):
        weights_names, weights_trainable, weights_values = \
            self.homomorphic_encryption.decrypt_pb_weights(model_pb)
        if len(self.weights_values) > 0:
            self.model_ops.get_model().set_model_weights(self.weights_names, self.weights_trainable, self.weights_values)
        return weights_names, weights_trainable, weights_values
    
    def _log(self, state, task):
        # FIXME:
        host_port = self.federation_helper.host_port_identifier()
        MetisLogger.info("Learner {} {} {} task on requested datasets."
                         .format(host_port, state, task))

    def evaluate_model(self, 
                        model_pb: model_pb2.Model, 
                        batch_size: int,
                        evaluation_datasets_pb: [learner_pb2.EvaluateModelRequest.dataset_to_eval],
                        metrics_pb: metis_pb2.EvaluationMetrics, 
                        verbose=False):       
        self._init_model_ops() 
        self._set_weights_from_model_pb(model_pb)
        
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
        metrics = [m for m in metrics_pb.metric]
        evaluation_datasets_pb = [d for d in evaluation_datasets_pb]

        train_eval = validation_eval = test_eval = dict()
        self._log(state="starts", task="evaluation")
        for dataset_to_eval in evaluation_datasets_pb:
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TRAINING:
                train_eval = self.model_ops.evaluate_model(train_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.VALIDATION:
                validation_eval = self.model_ops.evaluate_model(validation_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TEST:
                test_eval = self.model_ops.evaluate_model(test_dataset, batch_size, metrics, verbose)
        self._log(state="completed", task="evaluation")
        return self._get_completed_evaluation_task_pb(train_eval, validation_eval, test_eval)
 
    def infer_model(self, 
                    model_pb: model_pb2.Model, 
                    batch_size: int,
                    infer_train=False, 
                    infer_test=False, 
                    infer_valid=False, 
                    verbose=False):
        # TODO infer model should behave similarly as the evaluate_model(), by looping over a
        #  similar learner_pb2.InferModelRequest.dataset_to_infer list.
        self._init_model_ops()
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
        inferred_res = {
            "train": self.model_ops.infer_model(train_dataset, batch_size, verbose) if infer_train else None, 
            "valid": self.model_ops.infer_model(validation_dataset, batch_size, verbose) if infer_valid else None,
            "test": self.model_ops.infer_model(test_dataset, batch_size, verbose) if infer_test else None
        }            
        return DictionaryFormatter.stringify(inferred_res, stringify_nan=True)
        
    def train_model(self, 
                    model_pb: model_pb2.Model,
                    learning_task_pb, 
                    hyperparameters_pb,
                    verbose=False):
        self._init_model_ops()
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = self.learner_dataset.load_model_datasets()            
        self._log(state="starts", task="learning")
        model_weights_descriptor, learning_task_stats = self.model_ops.train_model(train_dataset, 
                                                            learning_task_pb, 
                                                            hyperparameters_pb,
                                                            validation_dataset, 
                                                            test_dataset, 
                                                            verbose)
        self._log(state="completed", task="learning")
        return  self._get_completed_learning_task_pb(model_weights_descriptor, learning_task_stats)

    def _get_completed_learning_task_pb(self, model_weights_descriptor, learning_task_stats):
        model_pb = self.homomorphic_encryption.encrypt_np_to_model_pb(model_weights_descriptor)
        completed_learning_task_pb = get_completed_learning_task_pb(
            model_pb=model_pb,
            learning_task_stats=learning_task_stats
        )
        return completed_learning_task_pb
    
    def _get_completed_evaluation_task_pb(self, train_eval, validation_eval, test_eval):
        return metis_pb2.ModelEvaluations(
            training_evaluation_pb=self._get_metric_pb(train_eval),
            validation_evaluation_pb=self._get_metric_pb(validation_eval),
            test_evaluation_pb=self._get_metric_pb(test_eval)
        )
        
    def _get_metric_pb(self, metrics):
        if not metrics:
            return metis_pb2.ModelEvaluation()
        metrics = DictionaryFormatter.stringify(metrics, stringify_nan=True)
        return metis_pb2.ModelEvaluation(metric_values=metrics)

    def __enter__(self):
        return self
 
    def __exit__(self):
        self.model_ops.cleanup()
