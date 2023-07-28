import tensorflow as tf
import tensorflow_addons as tfa

from metisfl.models.keras.keras_model import MetisModelKeras
from tensorflow.keras import regularizers


class AlzheimersDisease2DCNN(MetisModelKeras):

    # Model def
    def __init__(self, learning_rate=5e-5):
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91, 1), name='input')
        self.learning_rate = learning_rate
        super(AlzheimersDisease2DCNN, self).__init__()

    def get_model(self):
        x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same")(self.original_input)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(256, 3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.reduce_mean(x, axis=1)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        output = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

        model = tf.keras.Model(self.original_input, output, name="AD-2DCNN")
        the_metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        op = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss="binary_crossentropy", optimizer=op, metrics=the_metrics)

        return model


class AlzheimersDisease3DCNN(MetisModelKeras):

    # Model def
    def __init__(self, learning_rate=5e-5):
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91, 1), name='input')
        self.learning_rate = learning_rate
        super(AlzheimersDisease3DCNN, self).__init__()

    def get_model(self):
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, use_bias=False)(self.original_input)
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        # x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)

        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        # x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)

        x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        # x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)

        x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        # x = tfa.layers.InstanceNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)

        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

        model = tf.keras.Model(self.original_input, outputs, name="AD-3DCNN")
        the_metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        op = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
        # op = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-3)  # lambda = 0.001, 0.01, 0.1
        # op = tf.keras.optimizers.SGD(learning_rate=2e-4)
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=op, metrics=the_metrics)

        return model
