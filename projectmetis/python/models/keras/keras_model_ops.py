import numpy as np
import tensorflow as tf

from projectmetis.proto import model_pb2
from projectmetis.python.logging.metis_logger import MetisLogger
from projectmetis.python.models.keras.optimizers.fed_prox import FedProx
from projectmetis.python.models.model_dataset import ModelDataset
from projectmetis.python.models.model_ops import ModelOps
from projectmetis.python.models.keras.step_counter import StepCounter


class KerasModelOps(ModelOps):

    def __init__(self, model_filepath="/tmp/model", keras_callbacks=None, *args, **kwargs):
        if keras_callbacks is None:
            keras_callbacks = []
        elif len(keras_callbacks) > 0:
            valid_callbacks = any([isinstance(kc, tf.keras.callbacks.Callback) for kc in keras_callbacks])
            if not valid_callbacks:
                raise RuntimeError(
                    "{} needs to be an instance of {}".format(keras_callbacks, [tf.keras.callbacks.Callback]))
        self._model_filepath = model_filepath
        self._model = self.load_model(self._model_filepath)
        self._keras_callbacks = keras_callbacks
        super(KerasModelOps, self).__init__(self._model)

    def load_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Loading model from: {}".format(filepath))
        m = tf.keras.models.load_model(filepath)
        MetisLogger.info("Loaded model from: {}".format(filepath))
        return m

    def save_model(self, filepath=None, *args, **kwargs):
        if filepath is None:
            filepath = self._model_filepath
        MetisLogger.info("Saving model to: {}".format(filepath))
        # Save model in SavedModel format (default): https://www.tensorflow.org/guide/saved_model
        self._model.save(filepath=filepath)
        MetisLogger.info("Saved model at: {}".format(filepath))

    def set_model_weights(self, weights=None, *args, **kwargs):
        if weights is None:
            raise RuntimeError('Provided `weights` value is None.')
        else:
            MetisLogger.info("Applying new model weights")
            self._model.set_weights(weights)
            MetisLogger.info("Applied new model weights")

    def get_model_weights(self, *args, **kwargs):
        return self._model.get_weights()

    def train_model(self, dataset: ModelDataset = None, total_steps=100,
                    batch_size=100, verbose=False, *args, **kwargs):
        if dataset is None:
            raise RuntimeError("Provided `dataset` for training is None.")
        # Compute number of epochs based on the data size of the training set.
        dataset_size = dataset.get_size()
        steps_per_epoch = np.ceil(np.divide(dataset_size, batch_size))
        epochs_num = 1
        if total_steps > steps_per_epoch:
            epochs_num = np.divide(total_steps, steps_per_epoch)
        # Keras callback model training based on number of steps.
        step_counter_callback = StepCounter(total_steps=total_steps)
        MetisLogger.info("Starting model training.")
        # Keras does not accept halfway/floating epochs number,
        # hence the ceiling & integer conversion.
        # TODO batch_Size must be specified only when the input is a numpy array.
        #  For datasets/generators it should not be provided:
        #  https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        history_res = self._model.fit(dataset.get_x(), dataset.get_y(),
                                      epochs=np.int(np.ceil(epochs_num)), batch_size=batch_size,
                                      verbose=verbose, callbacks=[step_counter_callback, self._keras_callbacks])
        MetisLogger.info("Model training is complete.")
        # Since model has been changed, save the new model state.
        self.save_model(self._model_filepath)
        # `history_res` is an instance of keras.callbacks.History, hence the `.history` attribute.
        return history_res.history

    def evaluate_model(self, dataset: ModelDataset = None, batch_size=100,
                       verbose=False, metrics=None, *args, **kwargs):
        if dataset is None:
            raise RuntimeError("Provided `dataset` for evaluation is None.")
        MetisLogger.info("Starting model evaluation.")
        # TODO batch_Size must be specified only when the input is a numpy array.
        #  For datasets/generators it should not be provided:
        #  https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        eval_res = self._model.evaluate(dataset.get_x(), dataset.get_y(),
                                        batch_size=batch_size, callbacks=self._keras_callbacks,
                                        verbose=verbose, return_dict=True)
        MetisLogger.info("Model evaluation is complete.")
        return eval_res

    def infer_model(self, dataset: ModelDataset = None, batch_size=100, *args, **kwargs):
        if dataset is None:
            raise RuntimeError("Provided `dataset` for inference is None.")
        MetisLogger.info("Starting model inference.")
        predictions = self._model.predict(dataset.get_x(), batch_size, callbacks=self._keras_callbacks)
        MetisLogger.info("Model inference is complete.")
        return predictions

    def construct_optimizer(self, optimizer_config_pb: model_pb2.OptimizerConfig = None,
                            *args, **kwargs):
        if optimizer_config_pb is None:
            raise RuntimeError("Provided `OptimizerConfig` proto message is None.")
        if optimizer_config_pb.HasField('vanilla_sgd'):
            learning_rate = optimizer_config_pb.vanilla_sgd.learning_rate
            # TODO We might have to implement our own custom SGD with L2/L1, since Keras does not add L2 or L1
            #  regularization directly in the optimization function, it does so during model compilation
            #  at the kernel and bias level.
            l1_reg = optimizer_config_pb.vanilla_sgd.L1_reg
            l2_reg = optimizer_config_pb.vanilla_sgd.L2_reg
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, name='SGD')
        elif optimizer_config_pb.HasField('momentum_sgd'):
            learning_rate = optimizer_config_pb.momentum_sgd.learning_rate
            momentum_factor = optimizer_config_pb.momentum_sgd.momentum_factor
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum_factor, name='MomentumSGD')
        elif optimizer_config_pb.HasField('fed_prox'):
            learning_rate = optimizer_config_pb.fed_prox.learning_rate
            proximal_term = optimizer_config_pb.fed_prox.proximal_term
            return FedProx(learning_rate, proximal_term, name="FedProx")
        elif optimizer_config_pb.HasField('adam'):
            learning_rate = optimizer_config_pb.adam.learning_rate
            beta_1 = optimizer_config_pb.adam.beta_1
            beta_2 = optimizer_config_pb.adam.beta_2
            epsilon = optimizer_config_pb.adam.epsilon
            return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                                            amsgrad=False, name='Adam')
        else:
            raise RuntimeError("TrainingHyperparameters proto message refers to a non-supported optimizer.")
