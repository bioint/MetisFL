import tensorflow as tf

from keras import backend as K
from typing import Any, Dict, Tuple

from metisfl.proto import metis_pb2, model_pb2
from metisfl.models.keras.keras_model import MetisModelKeras
from metisfl.models.keras.callbacks.step_counter import StepCounter
from metisfl.models.keras.callbacks.performance_profiler import PerformanceProfiler
from metisfl.models.utils import calc_mean_wall_clock, get_num_of_epochs
from metisfl.models.types import LearningTaskStats, ModelWeightsDescriptor
from metisfl.common.logger import MetisLogger
from metisfl.common.formatting import DataTypeFormatter


class KerasModelOps(ModelOps):

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._set_gpu_memory_growth()
        self._metis_model = MetisModelKeras.load(model_dir)

    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False) -> Tuple[ModelWeightsDescriptor, LearningTaskStats]:
        if train_dataset is None:
            MetisLogger.fatal("Provided `dataset` for training is None.")
        MetisLogger.info("Starting model training.")

        global_iteration = learning_task_pb.global_iteration
        total_steps = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        dataset_size = train_dataset.get_size()
        step_counter_callback = StepCounter(total_steps=total_steps)
        performance_cb = PerformanceProfiler()
        self._construct_optimizer(hyperparameters_pb.optimizer)

        # @stripeli why is the epoch number calculated? Isn't it given in the yaml?
        # @stripeli why there are two batch sizes?
        epochs_num = get_num_of_epochs(
            dataset_size=dataset_size, batch_size=batch_size, total_steps=total_steps)
        x_train, y_train = train_dataset.construct_dataset_pipeline(
            batch_size=batch_size, is_train=True)
        x_valid, y_valid = validation_dataset.construct_dataset_pipeline(
            batch_size=batch_size, is_train=False)
        x_test, y_test = test_dataset.construct_dataset_pipeline(
            batch_size=batch_size, is_train=False)

        # We assign x_valid, y_valid only if both values
        # are not None, else we assign x_valid (None or not None).
        validation_data = (x_valid, y_valid) \
            if x_valid is not None and y_valid is not None else x_valid

        # Keras does not accept halfway/floating epochs number
        history_res = self._metis_model._backend_model.fit(x=x_train,
                                                           y=y_train,
                                                           batch_size=batch_size,
                                                           validation_data=validation_data,
                                                           epochs=epochs_num,
                                                           verbose=verbose,
                                                           callbacks=[step_counter_callback,
                                                                      performance_cb])

        # TODO(dstripelis) We evaluate the local model over the test dataset at the end of training.
        #  Maybe we need to parameterize evaluation at every epoch or at the end of training.
        # x_test could be an array - we just need to check if it is not None
        if x_test is not None:
            test_res = self._metis_model._backend_model.evaluate(x=x_test,
                                                                 y=y_test,
                                                                 batch_size=batch_size,
                                                                 verbose=verbose,
                                                                 return_dict=True)

        # Since model has been changed, save the new model state.
        self._metis_model.save(self._model_dir)
        # `history_res` is an instance of keras.callbacks.History, hence the `.history` attribute.
        training_res = {k: k_v for k, k_v in sorted(
            history_res.history.items()) if 'val_' not in k}
        # Replacing "val" so that metric keys are the same for both training and validation datasets.
        validation_res = {k.replace("val_", ""): k_v for k, k_v in sorted(
            history_res.history.items()) if 'val_' in k}

        model_weights_descriptor = self._metis_model.get_weights_descriptor()
        learning_task_stats = LearningTaskStats(
            train_stats=training_res,
            completed_epochs=epochs_num,
            global_iteration=global_iteration,
            validation_stats=validation_res,
            test_stats=test_res,
            completes_batches=total_steps,
            batch_size=batch_size,
            processing_ms_per_epoch=calc_mean_wall_clock(
                performance_cb.epochs_wall_clock_time_sec),
            processing_ms_per_batch=calc_mean_wall_clock(
                performance_cb.batches_wall_clock_time_sec)
        )
        MetisLogger.info("Model training is complete.")
        return model_weights_descriptor, learning_task_stats

    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       verbose=False) -> Dict:
        if eval_dataset is None:
            MetisLogger.fatal("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        x_eval, y_eval = eval_dataset.construct_dataset_pipeline(
            batch_size=batch_size, is_train=False)
        evaluation_metrics = dict()
        # x_eval, y_eval could be arrays - we just need to check if they are not None
        if x_eval is not None and y_eval is not None:
            evaluation_metrics = self._metis_model._backend_model.evaluate(x=x_eval,
                                                                           y=y_eval,
                                                                           batch_size=batch_size,
                                                                           verbose=verbose,
                                                                           return_dict=True)
        MetisLogger.info("Model evaluation is complete.")
        return evaluation_metrics

    def _set_gpu_memory_growth(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                MetisLogger.info("Physical GPUs: {}, Logical GPUs: {}".format(
                    len(gpus), len(logical_gpus)))
            except RuntimeError as e:
                MetisLogger.error(e)

    def _construct_optimizer(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        if optimizer_config_pb is None:
            MetisLogger.fatal(
                "Provided `OptimizerConfig` proto message is None.")
        params = optimizer_config_pb.params
        # We do this transformation to convert the optimizer parameters to snake case.
        # For instance: `LearningRate` -> `learning_rate`
        params = DataTypeFormatter.camel_to_snake_dict_keys(params)
        # We do not pick the optimizer using `tf.keras.optimizers.get(name)`
        # because the user might have defined an optimizer that is not part of
        # the standard Keras library. Therefore, we access directly the optimizer
        # of the defined model.
        optimizer = self._metis_model._backend_model.optimizer
        for param_name, param_val in params.items():
            for attr_name, attr_val in optimizer.__dict__.items():
                if param_name in attr_name:
                    attr_type = type(attr_val)
                    try:
                        # Need to see if we can cast the given value to
                        # the data type of the model's optimizer variable.
                        # If we can, then we assign the corresponding value.
                        param_val = (attr_type)(param_val)
                        if isinstance(optimizer, tf.keras.optimizers.legacy.Optimizer):
                            # When using the older Optimizer versions we simply
                            # assign the new value to the optimizer's attribute.
                            setattr(optimizer, attr_name, param_val)
                            # optimizer._learning_rate = param_val
                        else:
                            # When using the newer Optimizer versions we need to use
                            # the Keras Backend to set the corresponding value.
                            K.set_value(
                                getattr(optimizer, attr_name), param_val)
                    except Exception as e:
                        MetisLogger.warning(e)
