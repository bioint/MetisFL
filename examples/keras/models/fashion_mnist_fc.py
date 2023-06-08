import tensorflow as tf

from src.python.learner.models.model_def import ModelDef


class FashionMnistModel(ModelDef):

    def __init__(self, learning_rate=0.02):
        super(FashionMnistModel, self).__init__()
        self.learning_rate = learning_rate

    def get_model(self):
        """Prepare a simple dense model."""
        Dense = tf.keras.layers.Dense
        Flatten = tf.keras.layers.Flatten

        model = tf.keras.models.Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation="relu", kernel_initializer="glorot_uniform"))
        model.add(Dense(128, activation="relu", kernel_initializer="glorot_uniform"))
        model.add(Dense(10, activation="softmax", kernel_initializer="glorot_uniform"))
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model
