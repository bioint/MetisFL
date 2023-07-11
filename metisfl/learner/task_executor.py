from typing import Callable, List

from metisfl.encryption.homomorphic import Homomorphic
from metisfl.models.model_ops import ModelOps
from metisfl.models.utils import get_completed_learning_task_pb
from metisfl.proto import learner_pb2, metis_pb2, model_pb2
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.metis_logger import MetisLogger

from .dataset_handler import LearnerDataset


class TaskExecutor(object):

    def __init__(self, 
                 he_scheme_pb: metis_pb2.HESchemeConfig,
                 learner_dataset: LearnerDataset,
                 learner_server_entity_pb: metis_pb2.ServerEntity,
                 model_dir: str,
                 model_ops_fn: Callable[[str], ModelOps]):
        """A class that executes training/evaluation/inference tasks. The tasks in this class are
            executed in a independent process, different from the process that created the object. 
            It is importart to call the init_model_backend() method before calling any other method. 
            And it has to be called within the same process that runs the tasks so that the model
            backend is imported correctly.    

        Args:
            he_scheme_pb (metis_pb2.HESchemeConfig): A protobuf message that contains the HE scheme.
            learner_dataset (LearnerDataset): A LearnerDataset object that contains the datasets.
            model_backend_fn (Callable[[str], model_ops.ModelOps]): A function that returns a model backend.
            model_dir (str): The directory where the model is stored.
        """
        self._he_scheme_pb = he_scheme_pb
        self._learner_dataset = learner_dataset
        self._learner_server_entity_pb = learner_server_entity_pb
        self._model_ops = None 
        self._model_ops_fn = model_ops_fn
        self._model_dir = model_dir
        
    def _init_model_ops(self) -> ModelOps:
        if not self._model_ops:
            self._model_ops = self._model_ops_fn(self._model_dir)
            
    # @stripeli metrics_pb was not used anywhere, removed it
    def evaluate_model(self, 
                        model_pb: model_pb2.Model, 
                        batch_size: int,
                        evaluation_dataset_pb: List[learner_pb2.EvaluateModelRequest.dataset_to_eval],
                        verbose=False):       
        self._init_model_ops() 
        self._set_weights_from_model_pb(model_pb)
        
        train_dataset, validation_dataset, test_dataset = self._learner_dataset.load_model_datasets()

        train_eval = validation_eval = test_eval = dict()
        self._log(state="starts", task="evaluation")

        for dataset_to_eval in evaluation_dataset_pb:
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TRAINING:
                train_eval = self._model_ops.evaluate_model(train_dataset, batch_size, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.VALIDATION:
                validation_eval = self._model_ops.evaluate_model(validation_dataset, batch_size, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TEST:
                test_eval = self._model_ops.evaluate_model(test_dataset, batch_size, verbose)
                
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
            self._learner_dataset.load_model_datasets()
        inferred_res = {
            "train": self._model_ops.infer_model(train_dataset, batch_size, verbose) if infer_train else None, 
            "valid": self._model_ops.infer_model(validation_dataset, batch_size, verbose) if infer_valid else None,
            "test": self._model_ops.infer_model(test_dataset, batch_size, verbose) if infer_test else None
        }            
        return DictionaryFormatter.stringify(inferred_res, stringify_nan=True)
        
    def train_model(self, 
                    model_pb: model_pb2.Model,
                    learning_task_pb, 
                    hyperparameters_pb,
                    verbose=False):
        self._init_model_ops()
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = self._learner_dataset.load_model_datasets()            
        self._log(state="starts", task="learning")
        model_weights_descriptor, learning_task_stats = self._model_ops.train_model(train_dataset, 
                                                            learning_task_pb, 
                                                            hyperparameters_pb,
                                                            validation_dataset, 
                                                            test_dataset, 
                                                            verbose)
        self._log(state="completed", task="learning")
        return  self._get_completed_learning_task_pb(model_weights_descriptor, learning_task_stats)

    def _get_he_obj(self):
        return Homomorphic(self._he_scheme_pb)

    def _set_weights_from_model_pb(self, model_pb: model_pb2.Model):
        homomorphic_encryption = self._get_he_obj()
        model_weights_descriptor = homomorphic_encryption.decrypt_pb_weights(model_pb.variables)
        if len(model_weights_descriptor.weights_values) > 0:
            self._model_ops.get_model().set_model_weights(model_weights_descriptor)

    def _get_completed_learning_task_pb(self, model_weights_descriptor, learning_task_stats):
        homomorphic_encryption = self._get_he_obj()
        variables = homomorphic_encryption.encrypt_np_weights(model_weights_descriptor)
        model_pb = model_pb2.Model(variables=variables)
        completed_learning_task_pb = get_completed_learning_task_pb(
            model_pb=model_pb,
            learning_task_stats=learning_task_stats)
        return completed_learning_task_pb
    
    def _get_completed_evaluation_task_pb(self, train_eval, validation_eval, test_eval):
        return metis_pb2.ModelEvaluations(
            training_evaluation=self._get_metric_pb(train_eval),
            validation_evaluation=self._get_metric_pb(validation_eval),
            test_evaluation=self._get_metric_pb(test_eval))
        
    def _get_metric_pb(self, metrics):
        if not metrics:
            return metis_pb2.ModelEvaluation()
        metrics = DictionaryFormatter.stringify(metrics, stringify_nan=True)
        return metis_pb2.ModelEvaluation(metric_values=metrics)

    def _log(self, state, task):
        # FIXME:
        host_port = self._host_port_identifier()
        MetisLogger.info("Learner {} {} {} task on requested datasets."
                         .format(host_port, state, task))

    def _host_port_identifier(self):
        return "{}:{}".format(
            self._learner_server_entity_pb.hostname,
            self._learner_server_entity_pb.port)

    def __enter__(self):
        return self
 
    def __exit__(self):
        self._model_ops.cleanup()
