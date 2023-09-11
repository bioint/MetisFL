import tensorflow as tf

from metisfl.models.keras.keras_model import MetisModelKeras


class MelanomaFC(MetisModelKeras):

    def __init__(self, IMAGE_SIZE=[1024, 1024]):
        self.IMAGE_SIZE = IMAGE_SIZE

    def get_model(self):
        base_model = tf.keras.applications.Xception(
            input_shape=(*self.IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        inputs = tf.keras.layers.Input([*self.IMAGE_SIZE, 3])
        x = tf.keras.applications.xception.preprocess_input(inputs)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.7)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=tf.keras.metrics.AUC(name="auc"),
        )

        return model
