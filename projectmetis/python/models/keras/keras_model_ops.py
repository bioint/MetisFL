import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from projectmetis.proto import model_pb2
from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.models.keras.optimizers.fed_prox import FedProx
from projectmetis.python.models.keras.keras_proto_factory import KerasProtoFactory
from projectmetis.python.models.model_dataset import ModelDataset
from projectmetis.python.models.model_ops import ModelOps
from projectmetis.python.models.keras.callbacks.step_counter import StepCounter
from projectmetis.python.models.keras.callbacks.performance_profiler import PerformanceProfiler


class KerasModelOps(ModelOps):

    def __init__(self, model_filepath="/tmp/model", he_scheme=None, keras_callbacks=None, *args, **kwargs):
        # Runtime memory growth configuration for Tensorflow/Keras sessions.
        # Assumption is that the visible GPUs are set through the environmental
        # variable CUDA_VISIBLE_DEVICES, else it will consume all available GPUs.
        # The following code snippet is an official Tensorflow recommendation.
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

        if keras_callbacks is None:
            keras_callbacks = []
        elif len(keras_callbacks) > 0:
            are_callbacks_valid = any([isinstance(kc, tf.keras.callbacks.Callback) for kc in keras_callbacks])
            if not are_callbacks_valid:
                raise RuntimeError(
                    "{} needs to be an instance of {}".format(keras_callbacks, [tf.keras.callbacks.Callback]))

        self._model_filepath = model_filepath
        # TODO Register custom objects, e.g., optimizers, required to load the model.
        self._load_model_custom_objects = {"FedProx": FedProx}
        self._model = self.load_model(self._model_filepath)
        self._he_scheme = he_scheme
        self._keras_callbacks = keras_callbacks
        super(KerasModelOps, self).__init__(self._model, self._he_scheme)

    def _construct_dataset_pipeline(self, dataset: ModelDataset, batch_size, is_train=False):
        """
        A helper function to distinguish whether we have a tf.dataset or other data input sequence (e.g. numpy).
        We need to set up appropriately the data pipeline since keras method invocations require different parameters
        to be explicitly set. See also: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        :param dataset:
        :param batch_size:
        :param is_train:
        :return:
        """
        _x, _y, _b = None, None, None
        if dataset is not None and dataset.get_dataset() is not None and \
                isinstance(dataset.get_dataset(), tf.data.Dataset):
            _x = dataset.get_dataset()
            if is_train:
                # Shuffle all records only if dataset is used for training.
                _x = _x.shuffle(dataset.get_size())
            # If the input is of tf.Dataset we only need to return the input x,
            # we do not need to set a value for the rest of the variables (i.e., _y, _b).
            _x = _x.batch(batch_size)
        elif dataset.get_x() is not None and dataset.get_y() is not None:
            # This condition is reached when the input is of a numpy data type.
            # For this reason, we need to return both the x, y and the original batch size.
            _x, _y, _b = dataset.get_x(), dataset.get_y(), batch_size
        return _x, _y, _b

    def cleanup(self):
        del self._model
        tf.keras.backend.clear_session()

    def load_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Loading model from: {}".format(filepath))
        m = tf.keras.models.load_model(filepath, custom_objects=self._load_model_custom_objects)
        MetisLogger.info("Loaded model from: {}".format(filepath))
        return m

    def save_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Saving model to: {}".format(filepath))
        # Save model in SavedModel format (default): https://www.tensorflow.org/guide/saved_model
        self._model.save(filepath=filepath)
        MetisLogger.info("Saved model at: {}".format(filepath))

    def set_model_weights(self, weights_names, weights_trainable, weights_values, *args, **kwargs):
        MetisLogger.info("Applying new model weights")
        existing_weights = self._model.weights
        trainable_vars_names = [v.name for v in self._model.trainable_variables]
        assigning_weights = []
        for existing_weight, new_weight in zip(existing_weights, weights_values):
            # TODO It seems that it is better to assign the incoming model weight altogether.
            #  In a more fine grained implementation we should know whether to share all weights
            #  with the federation or a subset. This should be defined during initialization.
            assigning_weights.append(new_weight)
            # if existing_weight.name not in trainable_vars_names:
            #     assigning_weights.append(existing_weight.numpy())  # get the numpy/array values
            # else:
            #     assigning_weights.append(new_weight)
        self._model.set_weights(assigning_weights)
        MetisLogger.info("Applied new model weights")

    def get_model_weights(self, *args, **kwargs):
        return self._model.weights

    def train_model(self, train_dataset: ModelDataset, learning_task_pb,
                    hyperparameters_pb, validation_dataset: ModelDataset = None,
                    test_dataset: ModelDataset = None, verbose=False, *args, **kwargs):
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
            epochs_num = np.divide(total_steps, steps_per_epoch)
        # Keras callback model training based on number of steps.
        step_counter_callback = StepCounter(total_steps=total_steps)
        performance_profile_callback = PerformanceProfiler()
        MetisLogger.info("Starting model training.")

        x_train, y_train, fit_batch_size = self._construct_dataset_pipeline(
            dataset=train_dataset, batch_size=batch_size, is_train=True)
        x_valid, y_valid, _ = self._construct_dataset_pipeline(
            dataset=validation_dataset, batch_size=batch_size, is_train=False)
        x_test, y_test, _ = self._construct_dataset_pipeline(
            dataset=test_dataset, batch_size=batch_size, is_train=False)

        # We assign x_valid, y_valid only if both values
        # are not None, else we assign x_valid (None or not None).
        validation_data = (x_valid, y_valid) \
            if x_valid is not None and y_valid is not None else x_valid

        # Keras does not accept halfway/floating epochs number,
        # hence the ceiling & integer conversion.
        history_res = self._model.fit(x=x_train, y=y_train, batch_size=fit_batch_size,
                                      validation_data=validation_data,
                                      epochs=np.int(np.ceil(epochs_num)),
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
        self.save_model(self._model_filepath)
        # `history_res` is an instance of keras.callbacks.History, hence the `.history` attribute.
        training_res = {k: k_v for k, k_v in sorted(history_res.history.items()) if 'val_' not in k}
        # Replacing "val" so that metric keys are the same for both training and validation datasets.
        validation_res = {k.replace("val_", ""): k_v for k, k_v in sorted(history_res.history.items()) if 'val_' in k}
        # TODO Currently we do not evaluate the locally trained model against the test set, since
        #  we need to figure out if the evaluation will happen within this function scope or outside.
        completed_learning_task = KerasProtoFactory.CompletedLearningTaskProtoMessage(
            model=self._model, train_stats=training_res, completed_epochs=epochs_num,
            global_iteration=global_iteration, validation_stats=validation_res,
            test_stats=test_res, completes_batches=total_steps,
            batch_size=batch_size, processing_ms_per_epoch=mean_epoch_wall_clock_time_ms,
            processing_ms_per_batch=mean_batch_wall_clock_time_ms)
        completed_learning_task_pb = completed_learning_task.construct_completed_learning_task_pb(
            he_scheme=self._he_scheme)
        return completed_learning_task_pb

    def evaluate_model(self, dataset: ModelDataset = None, batch_size=100,
                       verbose=False, metrics=None, *args, **kwargs):
        if dataset is None:
            raise RuntimeError("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        # Set up properly data feeding.
        x_eval, y_eval, batch_size = self._construct_dataset_pipeline(
            dataset=dataset, batch_size=batch_size, is_train=False)
        # We need to make sure that the input x_eval is not None, since the `evaluate`
        # method raises error when x, y are None. To replicate the error, simply call
        # evaluate() with no arguments. The receiving error when this occurs is:
        # `Failed to find data adapter that can handle input: <class 'NoneType'>, <class 'NoneType'>`.
        eval_res = dict()
        if x_eval is not None:
            eval_res = self._model.evaluate(x=x_eval, y=y_eval, batch_size=batch_size,
                                            callbacks=self._keras_callbacks,
                                            verbose=verbose, return_dict=True)
        MetisLogger.info("Model evaluation is complete.")
        model_evaluation_pb = KerasProtoFactory\
            .ModelEvaluationProtoMessage(eval_res).construct_model_evaluation_pb()
        return model_evaluation_pb

    def infer_model(self, dataset: ModelDataset = None, batch_size=100, *args, **kwargs):
        if dataset is None:
            raise RuntimeError("Provided `dataset` for inference is None.")
        MetisLogger.info("Starting model inference.")
        # Set up properly data feeding.
        x_infer, _, batch_size = self._construct_dataset_pipeline(
            dataset=dataset, batch_size=batch_size, is_train=False)
        # Similar to evaluate(), we need to make sure that the input x_infer
        # is not None, since the `predict` method raises error when x is None.
        predictions = None
        if x_infer is not None:
            predictions = self._model.predict(x_infer, batch_size, callbacks=self._keras_callbacks)
        MetisLogger.info("Model inference is complete.")
        return predictions

    def construct_optimizer(self, optimizer_config_pb: model_pb2.OptimizerConfig = None,
                            *args, **kwargs):
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
