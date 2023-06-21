import cloudpickle
import collections
import inspect
import os
import shutil
import torch
import gc

import numpy as np

from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.metis_logger import MetisLogger
from metisfl.models.model_dataset import ModelDataset
from metisfl.models.model_ops import ModelOps
from metisfl.models.model_def import PyTorchDef
from metisfl.models.model_proto_factory import ModelProtoFactory
from metisfl.utils.formatting import DictionaryFormatter
from metisfl.utils.proto_messages_factory import MetisProtoMessages
from typing import List


class PyTorchModelOps(ModelOps):

    def __init__(self, model_dir="/tmp/metis/", he_scheme=None, *args, **kwargs):
        self._model_dir = model_dir
        self._model_weights_path = os.path.join(self._model_dir, "model_weights.pt")
        self._model_def_path = os.path.join(self._model_dir, "model_def.pkl")
        self._model = self.load_model(self._model_dir)
        self._he_scheme = he_scheme
        super(PyTorchModelOps, self).__init__(self._model, self._he_scheme)
        # self._model.to(device)
        # # pylint: disable=no-member
        # DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # m.to(DEVICE)

    def _construct_dataset_pipeline(self, dataset: ModelDataset):
        _x = dataset.get_x()
        _y = dataset.get_y()
        if _x and _y:
            return _x, _y
        elif _x:
            return _x
        else:
            MetisLogger.error("Not a well-formatted input dataset: {}, {}".format(_x, _y))
            return None

    def cleanup(self):
        del self._model
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, model_dir=None, *args, **kwargs):
        if model_dir is None:
            model_dir = self._model_dir
        MetisLogger.info("Loading model from: {}".format(model_dir))
        model_loaded = cloudpickle.load(open(self._model_def_path, "rb"))
        model_loaded.load_state_dict(torch.load(self._model_weights_path))
        MetisLogger.info("Loaded model from: {}".format(model_dir))
        return model_loaded

    def save_model(self, model_dir=None, *args, **kwargs):
        if model_dir is None:
            model_dir = self._model_dir
        MetisLogger.info("Saving model to: {}".format(model_dir))
        shutil.rmtree(self._model_dir)
        os.makedirs(self._model_dir)
        cloudpickle.register_pickle_by_value(inspect.getmodule(self._model))
        cloudpickle.dump(obj=self._model, file=open(self._model_def_path, "wb+"))
        torch.save(self._model.state_dict(), self._model_weights_path)
        MetisLogger.info("Saved model at: {}".format(model_dir))

    def set_model_weights(self,
                          weights_names: List[str],
                          weights_trainable: List[bool],
                          weights_values: List[np.ndarray],
                          *args, **kwargs):
        state_dict = collections.OrderedDict({
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(self._model.state_dict().keys(), weights_values)
        })
        self._model.load_state_dict(state_dict, strict=True)

    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False,
                    *args, **kwargs) -> metis_pb2.CompletedLearningTask:

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
        dataset = self._construct_dataset_pipeline(train_dataset)

        fit_fn = getattr(self._model, "fit", None)
        # We need to make sure that the given dataset is not None and that the
        # given fit function implements the abstract fit method of PyTorchDef
        # class, and of course make sure that the function itself is callable.
        train_res = {}
        if dataset:
            if isinstance(self._model, PyTorchDef) and callable(fit_fn):
                MetisLogger.info("Using provided fit function.")
                train_res = self._model.fit(dataset, epochs=epochs_num)
            else:
                MetisLogger.error("Fit function not provided, please implement one.")

        MetisLogger.info("Model training is complete.")

        model_weights_descriptor = self.get_model_weights()
        # TODO (dstripelis) Need to add the metrics for computing the execution time
        #   per batch and epoch.
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
                       verbose=False,
                       *args, **kwargs) -> metis_pb2.ModelEvaluation:

        MetisLogger.info("Starting model evaluation.")
        dataset = self._construct_dataset_pipeline(eval_dataset)
        # Set model to evaluation state.
        self._model.eval()
        evaluate_fn = getattr(self._model, "evaluate", None)
        # We need to make sure that the given dataset is not None and that the given
        # evaluate function implements the abstract evaluate method of PyTorchDef class,
        # and of course make sure that the function itself is callable.
        eval_res = {}
        if dataset:
            if isinstance(self._model, PyTorchDef) and callable(evaluate_fn):
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
