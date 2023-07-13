import tensorflow as tf
import numpy as np

from metisfl import config
from metisfl.models.keras.optimizers.fed_prox import FedProx
from metisfl.models.metis_model import MetisModel
from metisfl.models.types import ModelWeightsDescriptor
from metisfl.utils.metis_logger import MetisLogger


class MetisModelKeras(MetisModel):

    def __init__(self, model: tf.keras.Model):
        assert isinstance(
            model, tf.keras.Model), "Model must be a tf.keras.Model instance."
        self._backend_model = model
        self._nn_engine = config.KERAS_NN_ENGINE

    @staticmethod
    def load(model_dir) -> "MetisModelKeras":
        MetisLogger.info("Loading model from: {}".format(model_dir))

        m = tf.keras.models.load_model(
            model_dir, custom_objects={"FedProx": FedProx})

        MetisLogger.info("Loaded model from: {}".format(model_dir))
        return MetisModelKeras(m)

    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        weights_names, weights_trainable, weights_values = [], [], []

        weights_names = [
            w.name for layer in self._backend_model.layers for w in layer.weights]

        all_trainable_weights_names = [
            v.name for v in self._backend_model.trainable_variables]

        weights_trainable = [
            True if w_n in all_trainable_weights_names else False for w_n in weights_names]

        weights_values = [w.numpy() for w in self._backend_model.weights]

        return ModelWeightsDescriptor(weights_names=weights_names,
                                      weights_trainable=weights_trainable,
                                      weights_values=weights_values)

    def save(self, model_dir, initialize=False) -> None:
        if initialize:
            self._run_initial_evaluation()
        MetisLogger.info("Saving model to: {}".format(model_dir))
        self._backend_model.save(filepath=model_dir)
        MetisLogger.info("Saved model at: {}".format(model_dir))

    def set_model_weights(self, model_weights_descriptor: ModelWeightsDescriptor):
        MetisLogger.info("Applying new model weights")
        weights_values = model_weights_descriptor.weights_values
        existing_weights = self._backend_model.weights
        trainable_vars_names = [
            v.name for v in self._backend_model.trainable_variables]
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
        self._backend_model.set_weights(assigning_weights)
        MetisLogger.info("Applied new model weights")

    def _run_initial_evaluation(self) -> None:
        # FIXME: This is a hack; will only work for 1-dim output
        input_shape = self._backend_model.layers[0].input_shape
        output_shape = self._backend_model.layers[-1].output_shape
        input_shape = (1, *input_shape[1:])
        output_shape = (1,)
        x = np.random.random(input_shape)
        y = np.random.random(output_shape)
        self._backend_model.evaluate(x, y, verbose=0)
