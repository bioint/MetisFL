import gc

import torch
import numpy as np

from metisfl.models.keras.helper import construct_dataset_pipeline
from metisfl.models.model_dataset import ModelDataset
from metisfl.models.model_ops import ModelOps
from metisfl.models.model_proto_factory import ModelProtoFactory
from metisfl.models.pytorch.wrapper import MetisTorchModel
from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.metis_logger import MetisLogger
from metisfl.utils.proto_messages_factory import MetisProtoMessages

class PyTorchModelOps(ModelOps):
    
    # @stripeli - we must remove default params like this one
    # bacause we might forget to put it when we instantiate the class
    def __init__(self, model_dir="/tmp/metis/"):
        self._model = MetisTorchModel().load(model_dir)
        # self._model.to(device)
        # # pylint: disable=no-member
        # @stripeli - I think it should be up to the user to decide whether to use GPU or not; 
        # DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # m.to(DEVICE)

    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False) -> metis_pb2.CompletedLearningTask:

        global_iteration = learning_task_pb.global_iteration
        total_steps = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        dataset_size = train_dataset.get_size()
        steps_per_epoch = np.ceil(np.divide(dataset_size, batch_size))
        epochs_num = 1
        if total_steps > steps_per_epoch:
            epochs_num = int(np.ceil(np.divide(total_steps, steps_per_epoch)))

        MetisLogger.info("Starting model training.")
        # Set model to train state.
        self._model.train()
        dataset = construct_dataset_pipeline(train_dataset)

        fit_fn = getattr(self._model, "fit", None)
        # We need to make sure that the given dataset is not None and that the
        # given fit function implements the abstract fit method of TorchModelDef
        # class, and of course make sure that the function itself is callable.
        train_res = {}
        if dataset:
            if isinstance(self._model, MetisTorchModel) and callable(fit_fn):
                MetisLogger.info("Using provided fit function.")
                train_res = self._model.fit(dataset, epochs=epochs_num)
            else:
                MetisLogger.error("Fit function not provided, please implement one.")

        MetisLogger.info("Model training is complete.")

        model_weights_descriptor = self.get_model_weights()
        # TODO (dstripelis) Need to add the metrics for computing the execution time
        #   per batch and epoch.
        # @stripeli no need unpack model_weights_descriptor
        completed_learning_task = ModelProtoFactory.CompletedLearningTaskProtoMessage(
            weights_values=model_weights_descriptor.weights_values,
            weights_trainable=model_weights_descriptor.weights_trainable,
            weights_names=model_weights_descriptor.weights_names,
            train_stats=train_res,
            completed_epochs=epochs_num,
            global_iteration=learning_task_pb.global_iteration)
        completed_learning_task_pb = completed_learning_task.construct_completed_learning_task_pb(
            he_scheme=self._he_scheme)
        return completed_learning_task_pb

    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       metrics=None,
                       verbose=False) -> metis_pb2.ModelEvaluation:

        MetisLogger.info("Starting model evaluation.")
        dataset = construct_dataset_pipeline(eval_dataset)
        # Set model to evaluation state.
        self._model.eval()
        evaluate_fn = getattr(self._model, "evaluate", None)
        # We need to make sure that the given dataset is not None and that the given
        # evaluate function implements the abstract evaluate method of TorchModelDef class,
        # and of course make sure that the function itself is callable.
        eval_res = {}
        if dataset:
            if isinstance(self._model, MetisTorchModel) and callable(evaluate_fn):
                MetisLogger.info("Using provided fit function.")
                eval_res = self._model.evaluate(dataset)
            else:
                MetisLogger.error("Evaluate function not provided, please implement one.")

        MetisLogger.info("Model evaluation is complete.")
        metric_values = DictionaryFormatter.stringify(eval_res, stringify_nan=True)
        return MetisProtoMessages.construct_model_evaluation_pb(metric_values)

    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100,
                    *args, **kwargs):

        # Set model to evaluation state.
        self._model.eval()
        pass

    def construct_optimizer(self,
                            optimizer_config_pb: model_pb2.OptimizerConfig = None,
                            *args, **kwargs):
        pass

    # @stripeli do we really need this?
    def cleanup(self):
        del self._model
        torch.cuda.empty_cache()
        gc.collect()
