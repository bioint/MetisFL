import metisfl.utils.proto_messages_factory as proto_factory
from metisfl.learner.dataset_handler import LearnerDataset
from metisfl.models.model_ops import ModelOps
from metisfl.proto import learner_pb2, model_pb2, metis_pb2
from metisfl.utils.fedenv_parser import HomomorphicEncryption
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.metis_logger import MetisLogger


class LearnerEvaluator(object):

    def __init__(self, 
                 homomorphic_encryption: HomomorphicEncryption,
                 learner_dataset: LearnerDataset,
                 model_backend: ModelOps):

        self.homomorphic_encryption = homomorphic_encryption
        self.learner_dataset = learner_dataset
        self.model_backend = model_backend
                        
    def _set_weights_from_model_pb(self, model_pb):
        weights_names, weights_trainable, weights_values = \
            self.homomorphic_encryption.decrypt_pb_weights(model_pb)
        if len(self.weights_values) > 0:
            self.model_backend.get_model().set_model_weights(self.weights_names, self.weights_trainable, self.weights_values)
        return weights_names, weights_trainable, weights_values

    def evaluate_model(self, 
                        model_pb: model_pb2.Model, 
                        batch_size: int,
                        evaluation_datasets_pb: [learner_pb2.EvaluateModelRequest.dataset_to_eval],
                        metrics_pb: metis_pb2.EvaluationMetrics, 
                        verbose=False):        
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
            
        # Need to unfold the pb into python list.
        metrics = [m for m in metrics_pb.metric]
                
        # Initialize to an empty metis_pb2.ModelEvaluation object all three variables.
        train_eval_pb = validation_eval_pb = test_eval_pb = metis_pb2.ModelEvaluation()
        for dataset_to_eval in evaluation_datasets_pb:
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TRAINING:
                train_eval_pb = self.model_backend.evaluate_model(train_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.VALIDATION:
                validation_eval_pb = self.model_backend.evaluate_model(validation_dataset, batch_size, metrics, verbose)
            if dataset_to_eval == learner_pb2.EvaluateModelRequest.dataset_to_eval.TEST:
                test_eval_pb = self.model_backend.evaluate_model(test_dataset, batch_size, metrics, verbose)
                                
        model_evaluations_pb = \
            proto_factory.MetisProtoMessages.construct_model_evaluations_pb(
                training_evaluation_pb=train_eval_pb,
                validation_evaluation_pb=validation_eval_pb,
                test_evaluation_pb=test_eval_pb)
        return model_evaluations_pb

    def infer_model(self, 
                    model_pb: model_pb2.Model, 
                    batch_size: int,
                    infer_train=False, 
                    infer_test=False, 
                    infer_valid=False, 
                    verbose=False):
        # TODO infer model should behave similarly as the evaluate_model(), by looping over a
        #  similar learner_pb2.InferModelRequest.dataset_to_infer list.
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()
        inferred_res = {
            "train": self.model_backend.infer_model(train_dataset, batch_size, verbose) if infer_train else None, 
            "valid": self.model_backend.infer_model(validation_dataset, batch_size, verbose) if infer_valid else None,
            "test": self.model_backend.infer_model(test_dataset, batch_size, verbose) if infer_test else None
        }            
        stringified_res = DictionaryFormatter.stringify(inferred_res, stringify_nan=True)
        return stringified_res
        
    def train_model(self, 
                    model_pb: model_pb2.Model,
                    learning_task_pb, 
                    hyperparameters_pb,
                    verbose=False):
        self._set_weights_from_model_pb(model_pb)
        train_dataset, validation_dataset, test_dataset = \
            self.learner_dataset.load_model_datasets()            
        completed_task_pb = self.model_backend\
            .train_model(train_dataset, learning_task_pb, hyperparameters_pb,
                         validation_dataset, test_dataset, verbose)
        
        MetisLogger.info("Learner {} completed model training on local training dataset."
                            .format(self.host_port_identifier()))
        return completed_task_pb

    def __enter__(self):
        return self

    def __exit__(self):
        self.model_backend.cleanup()

