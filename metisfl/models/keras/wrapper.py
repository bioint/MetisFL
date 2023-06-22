from typing import List
import tensorflow as tf
import numpy as np
from metisfl.models.model_wrapper import MetisModel

from metisfl.learner.weight_decrypt import ModelWeightsDescriptor
from metisfl.utils.metis_logger import MetisLogger

class MetisKerasModel(MetisModel):

    def __init__(self, model: tf.keras.Model):
        assert isinstance(model, tf.keras.Model), "Model must be a tf.keras.Model instance."
        self.model = model
        # nn_model.evaluate(x=np.random.random(x_train[0:1].shape), y=np.random.random(y_train[0:1].shape), verbose=False)
        self.nn_engine = "keras"

    def load(self, model_dir) -> tf.keras.Model:
        MetisLogger.info("Loading model from: {}".format(model_dir))
        m = tf.keras.models.load_model(model_dir, custom_objects=self._load_model_custom_objects)
        MetisLogger.info("Loaded model from: {}".format(model_dir))
        return m

    def get_weights_descriptor(self) -> ModelWeightsDescriptor:
        weights_names, weights_trainable, weights_values = [], [], []
        weights_names = [w.name for layer in self.model.layers for w in layer.weights]
        all_trainable_weights_names = [v.name for v in self.model.trainable_variables]
        weights_trainable = [True if w_n in all_trainable_weights_names else False for w_n in weights_names]
        weights_values = [w.numpy() for w in self.model.weights]
        return ModelWeightsDescriptor(nn_engine=self.nn_engine,
                                      weights_names=weights_names,
                                      weights_trainable=weights_trainable,
                                      weights_values=weights_values)
        
    def save(self, model_dir):
        MetisLogger.info("Saving model to: {}".format(model_dir))
        # Save model in SavedModel format (default): https://www.tensorflow.org/guide/saved_model
        self.model.save(filepath=model_dir)
        MetisLogger.info("Saved model at: {}".format(model_dir))
          
    def set_model_weights(self,
                          weights_names: List[str],
                          weights_trainable: List[bool],
                          weights_values: List[np.ndarray],
                          *args, **kwargs):
        MetisLogger.info("Applying new model weights")
        existing_weights = self.model.weights
        trainable_vars_names = [v.name for v in self.model.trainable_variables]
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
        self.model.set_weights(assigning_weights)
        MetisLogger.info("Applied new model weights")