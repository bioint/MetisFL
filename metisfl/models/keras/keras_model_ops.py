import numpy as np
import tensorflow as tf
from metisfl.models.keras.helper import construct_dataset_pipeline
from metisfl.models.model_ops import CompletedTaskStats, ModelOps

from metisfl.proto import metis_pb2, model_pb2
from metisfl.utils.metis_logger import MetisLogger
from metisfl.models.model_dataset import ModelDataset
from metisfl.models.keras.wrapper import MetisKerasModel
from metisfl.models.model_proto_factory import ModelProtoFactory
from metisfl.models.keras.callbacks.step_counter import StepCounter
from metisfl.models.keras.callbacks.performance_profiler import PerformanceProfiler


class KerasModelOps(ModelOps):

    def __init__(self, 
                 model_dir,
                 keras_callbacks=None):
        # Runtime memory growth configuration for Tensorflow/Keras sessions.
        # Assumption is that the visible GPUs are set through the environmental
        # variable CUDA_VISIBLE_DEVICES, else it will consume all available GPUs.
        # The following code snippet is an official Tensorflow recommendation.
        
        # @stripeli - why so many config for tensorflow and non for pytorch?
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                MetisLogger.info("Physical GPUs: {}, Logical GPUs: {}".format(len(gpus), len(logical_gpus)))
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized.
                MetisLogger.error(e)

        self._model_dir = model_dir
        # TODO Register custom objects, e.g., optimizers, required to load the model.
        self._model = MetisKerasModel.load(model_dir)
        self._keras_callbacks = keras_callbacks

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
        # TODO Compile model with new optimizer if need to.
        #  It is required by TF when redefining a model.
        #  Assign new model weights after model compilation.
        self.construct_optimizer(hyperparameters_pb.optimizer)
        if train_dataset is None:
            raise RuntimeError("Provided `dataset` for training is None.")
        # Compute number of epochs based on the data size of the training set.
        dataset_size = train_dataset.get_size()
        steps_per_epoch = np.ceil(np.divide(dataset_size, batch_size))
        epochs_num = 1
        if total_steps > steps_per_epoch:
            epochs_num = int(np.ceil(np.divide(total_steps, steps_per_epoch)))
        # Keras callback model training based on number of steps.
        step_counter_callback = StepCounter(total_steps=total_steps)
        performance_profile_callback = PerformanceProfiler()
        MetisLogger.info("Starting model training.")

        x_train, y_train = construct_dataset_pipeline(
            dataset=train_dataset, batch_size=batch_size, is_train=True)
        x_valid, y_valid = construct_dataset_pipeline(
            dataset=validation_dataset, batch_size=batch_size, is_train=False)
        x_test, y_test = construct_dataset_pipeline(
            dataset=test_dataset, batch_size=batch_size, is_train=False)

        # We assign x_valid, y_valid only if both values
        # are not None, else we assign x_valid (None or not None).
        validation_data = (x_valid, y_valid) \
            if x_valid is not None and y_valid is not None else x_valid

        # Keras does not accept halfway/floating epochs number,
        # hence the ceiling & integer conversion.
        history_res = self._model.fit(x=x_train,
                                      y=y_train,
                                      batch_size=batch_size,
                                      validation_data=validation_data,
                                      epochs=epochs_num,
                                      verbose=verbose,
                                      callbacks=[step_counter_callback,
                                                 performance_profile_callback,
                                                 self._keras_callbacks])

        # Compute mean wall clock time for epoch and batch.
        mean_epoch_wall_clock_time_ms = \
            np.mean(performance_profile_callback.epochs_wall_clock_time_sec) * 1000
        mean_batch_wall_clock_time_ms = \
            np.mean(performance_profile_callback.batches_wall_clock_time_sec) * 1000

        # TODO(dstripelis) We evaluate the local model over the test dataset at the end of training.
        #  Maybe we need to parameterize evaluation at every epoch or at the end of training.
        test_res = self._model.evaluate(x=x_test, y=y_test, batch_size=batch_size,
                                        callbacks=self._keras_callbacks,
                                        verbose=verbose, return_dict=True)

        MetisLogger.info("Model training is complete.")
        # Since model has been changed, save the new model state.
        self.save_model(self._model_dir)
        # `history_res` is an instance of keras.callbacks.History, hence the `.history` attribute.
        training_res = {k: k_v for k, k_v in sorted(history_res.history.items()) if 'val_' not in k}
        # Replacing "val" so that metric keys are the same for both training and validation datasets.
        validation_res = {k.replace("val_", ""): k_v for k, k_v in sorted(history_res.history.items()) if 'val_' in k}
        # TODO Currently we do not evaluate the locally trained model against the test set, since
        #  we need to figure out if the evaluation will happen within this function scope or outside.
        model_weights_descriptor = self._model.get_model_weights()
        
        comleted_task_stats = CompletedTaskStats(
            train_stats=training_res, 
            completed_epochs=epochs_num,
            global_iteration=global_iteration, 
            validation_stats=validation_res,
            test_stats=test_res, 
            completes_batches=total_steps,
            batch_size=batch_size, 
            processing_ms_per_epoch=mean_epoch_wall_clock_time_ms,
            processing_ms_per_batch=mean_batch_wall_clock_time_ms)
        # completed_learning_task_pb = completed_learning_task.construct_completed_learning_task_pb(
        #     he_scheme=self._he_scheme)
        return model_weights_descriptor, comleted_task_stats

    def evaluate_model(self,
                       eval_dataset: ModelDataset,
                       batch_size=100,
                       verbose=False) -> metis_pb2.ModelEvaluation:
        if eval_dataset is None:
            raise RuntimeError("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        # Set up properly data feeding.
        x_eval, y_eval = self._construct_dataset_pipeline(
            dataset=eval_dataset, batch_size=batch_size, is_train=False)
        # We need to make sure that the input x_eval is not None, since the `evaluate`
        # method raises error when x, y are None. To replicate the error, simply call
        # evaluate() with no arguments. The receiving error when this occurs is:
        # `Failed to find data adapter that can handle input: <class 'NoneType'>, <class 'NoneType'>`.
        # @stripeli: you are already making this check at the begining of the function.
        eval_res = dict()
        if x_eval is not None:
            eval_res = self._model.evaluate(x=x_eval,
                                            y=y_eval,
                                            batch_size=batch_size,
                                            callbacks=self._keras_callbacks,
                                            verbose=verbose, return_dict=True)
        MetisLogger.info("Model evaluation is complete.")
        # model_evaluation_pb = ModelProtoFactory\
        #     .ModelEvaluationProtoMessage(eval_res).construct_model_evaluation_pb()
        return eval_res

    def infer_model(self,
                    infer_dataset: ModelDataset,
                    batch_size=100):
        if infer_dataset is None:
            raise RuntimeError("Provided `dataset` for inference is None.")
        MetisLogger.info("Starting model inference.")
        # Set up properly data feeding.
        x_infer, _ = self._construct_dataset_pipeline(
            dataset=infer_dataset, batch_size=batch_size, is_train=False)
        # Similar to evaluate(), we need to make sure that the input x_infer
        # is not None, since the `predict` method raises error when x is None.
        predictions = None
        if x_infer is not None:
            predictions = self._model.predict(x_infer, batch_size, callbacks=self._keras_callbacks)
        MetisLogger.info("Model inference is complete.")
        return predictions

    def construct_optimizer(self, optimizer_config_pb: model_pb2.OptimizerConfig):
        if optimizer_config_pb is None:
            raise RuntimeError("Provided `OptimizerConfig` proto message is None.")
        if optimizer_config_pb.HasField('vanilla_sgd'):
            learning_rate = optimizer_config_pb.vanilla_sgd.learning_rate
            # TODO For now we only assign the learning rate, since Keras does not add L2 or L1
            #  regularization directly in the optimization function, it does so during model
            #  compilation at the kernel and bias layers level.
            l1_reg = optimizer_config_pb.vanilla_sgd.L1_reg
            l2_reg = optimizer_config_pb.vanilla_sgd.L2_reg
            self._model.optimizer.learning_rate.assign(learning_rate)
        elif optimizer_config_pb.HasField('momentum_sgd'):
            learning_rate = optimizer_config_pb.momentum_sgd.learning_rate
            momentum_factor = optimizer_config_pb.momentum_sgd.momentum_factor
            self._model.optimizer.learning_rate.assign(learning_rate)
            self._model.optimizer.momentum.assign(momentum_factor)
        elif optimizer_config_pb.HasField('fed_prox'):
            learning_rate = optimizer_config_pb.fed_prox.learning_rate
            proximal_term = optimizer_config_pb.fed_prox.proximal_term
            self._model.optimizer.learning_rate.assign(learning_rate)
            self._model.optimizer.proximal_term.assign(proximal_term)
        elif optimizer_config_pb.HasField('adam'):
            learning_rate = optimizer_config_pb.adam.learning_rate
            beta_1 = optimizer_config_pb.adam.beta_1
            beta_2 = optimizer_config_pb.adam.beta_2
            epsilon = optimizer_config_pb.adam.epsilon
            self._model.optimizer.learning_rate.assign(learning_rate)
            self._model.optimizer.beta_1.assign(beta_1)
            self._model.optimizer.beta_2.assign(beta_2)
            self._model.optimizer.epsilon.assign(epsilon)
        elif optimizer_config_pb.HasField('adam_weight_decay'):
            learning_rate = optimizer_config_pb.adam_weight_decay.learning_rate
            weight_decay = optimizer_config_pb.adam_weight_decay.weight_decay
            self._model.optimizer.learning_rate.assign(learning_rate)
            self._model.optimizer.weight_decay.assign(weight_decay)
        else:
            raise RuntimeError("TrainingHyperparameters proto message refers to a non-supported optimizer.")

    def cleanup(self):
        del self._model
        tf.keras.backend.clear_session()    
