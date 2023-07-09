from typing import Any

import tensorflow as tf

from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.metis_logger import MetisLogger
from metisfl.models.model_dataset import ModelDataset
from metisfl.models.keras.wrapper import MetisKerasModel
from metisfl.models.keras.callbacks.step_counter import StepCounter
from metisfl.models.keras.callbacks.performance_profiler import PerformanceProfiler
from metisfl.models.model_ops import LearningTaskStats, ModelOps
from metisfl.models.utils import calc_mean_wall_clock, get_num_of_epochs


class KerasModelOps(ModelOps):

    def __init__(self, model_dir: str):
        self._model_dir = model_dir
        self._set_gpu_memory_growth()
        self._metis_model = MetisKerasModel.load(model_dir)

    def train_model(self,
                    train_dataset: ModelDataset,
                    learning_task_pb: metis_pb2.LearningTask,
                    hyperparameters_pb: metis_pb2.Hyperparameters,
                    validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None,
                    verbose=False) -> metis_pb2.CompletedLearningTask:
        if train_dataset is None:
            raise RuntimeError("Provided `dataset` for training is None.")
        MetisLogger.info("Starting model training.")

        global_iteration = learning_task_pb.global_iteration
        total_steps = learning_task_pb.num_local_updates
        batch_size = hyperparameters_pb.batch_size
        dataset_size = train_dataset.get_size()
        step_counter_callback = StepCounter(total_steps=total_steps)
        performance_cb = PerformanceProfiler()
        
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
                       verbose=False) -> Any | dict:
        if eval_dataset is None:
            raise RuntimeError("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        x_eval, y_eval = eval_dataset.construct_dataset_pipeline(batch_size=batch_size, is_train=False)
        evaluation_metrics = self._metis_model._backend_model.evaluate(x=x_eval,
                                                                       y=y_eval,
                                                                       batch_size=batch_size,
                                                                       verbose=verbose, 
                                                                       return_dict=True)
        MetisLogger.info("Model evaluation is complete.")
        return evaluation_metrics

    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100) -> Any | None:
        if infer_dataset is None:
            raise RuntimeError("Provided `dataset` for inference is None.")
        MetisLogger.info("Starting model inference.")
        x_infer, _ = infer_dataset.construct_dataset_pipeline(
            batch_size=batch_size, is_train=False)
        predictions = self._metis_model._backend_model.predict(
            x_infer, batch_size)
        MetisLogger.info("Model inference is complete.")
        return predictions

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
            raise RuntimeError(
                "Provided `OptimizerConfig` proto message is None.")
        optim_fun = {
            "vanilla_sgd": self._construst_vanila_sgd,
            "momentum_sgd": self._construct_momentum_sgd,
            "fed_prox": self._construct_fed_prox,
            "adam": self._construct_adam,
            "adam_weight_decay": self._construct_adam_weight_decay
        }
        for optim, fun in optim_fun.items():
            if optimizer_config_pb.HasField(optim):
                return fun(optimizer_config_pb)

        raise RuntimeError(
            "TrainingHyperparameters proto message refers to a non-supported optimizer.")

    def _construst_vanila_sgd(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        learning_rate = optimizer_config_pb.vanilla_sgd.learning_rate
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def _construct_momentum_sgd(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        learning_rate = optimizer_config_pb.momentum_sgd.learning_rate
        momentum_factor = optimizer_config_pb.momentum_sgd.momentum_factor
        self._metis_model._backend_model.optimizer.learning_rate.assign(
            learning_rate)
        self._metis_model._backend_model.optimizer.momentum.assign(
            momentum_factor)  # FIXME: tf bug, mommentum is not a tf variable

    def _construct_fed_prox(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        learning_rate = optimizer_config_pb.fed_prox.learning_rate
        proximal_term = optimizer_config_pb.fed_prox.proximal_term
        self._metis_model._backend_model.optimizer.learning_rate.assign(
            learning_rate)
        self._metis_model._backend_model.optimizer.proximal_term.assign(
            proximal_term)

    def _construct_adam(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        learning_rate = optimizer_config_pb.adam.learning_rate
        beta_1 = optimizer_config_pb.adam.beta_1
        beta_2 = optimizer_config_pb.adam.beta_2
        epsilon = optimizer_config_pb.adam.epsilon
        self._metis_model = self._get_self._mode()
        self._metis_model._backend_model.optimizer.learning_rate.assign(
            learning_rate)
        self._metis_model._backend_model.optimizer.beta_1.assign(beta_1)
        self._metis_model._backend_model.optimizer.beta_2.assign(beta_2)
        self._metis_model._backend_model.optimizer.epsilon.assign(epsilon)

    def _construct_adam_weight_decay(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        learning_rate = optimizer_config_pb.adam_weight_decay.learning_rate
        weight_decay = optimizer_config_pb.adam_weight_decay.weight_decay
        self._metis_model._backend_model.optimizer.learning_rate.assign(
            learning_rate)
        self._metis_model._backend_model.optimizer.weight_decay.assign(
            weight_decay)

    def cleanup(self):
        del self._metis_model
        tf.keras.backend.clear_session()
