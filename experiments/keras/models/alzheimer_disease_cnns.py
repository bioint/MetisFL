import tensorflow as tf

from projectmetis.python.models.model_def import ModelDef


class AlzheimerDisease2DCNN(ModelDef):

    # Model def
    def __init__(self, learning_rate=5e-5, batch_size=1):
        # TODO Ask learning rate
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91), batch_size=batch_size, name='input')
        self.learning_rate = learning_rate
        super(AlzheimerDisease2DCNN, self).__init__()

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

        model = tf.keras.Model(self.original_input, output, name="2DCNN")
        the_metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        op = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss="binary_crossentropy", optimizer=op, metrics=the_metrics)

        return model


class AlzheimerDisease3DCNN(ModelDef):

    # Model def
    def __init__(self, learning_rate=5e-5, batch_size=1):
        # TODO Ask learning rate
        self.original_input = tf.keras.layers.Input(shape=(91, 109, 91, 1), batch_size=batch_size, name='input')
        self.learning_rate = learning_rate
        super(AlzheimerDisease3DCNN, self).__init__()

    def get_model(self):
        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(self.original_input)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        # TODO Ask dropout rate
        # x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

        model = tf.keras.Model(self.original_input, outputs, name="3DCNN")
        the_metrics = [
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        # TODO This needs to change to Adam with WeightDecay
        op = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss="binary_crossentropy", optimizer=op, metrics=the_metrics)

        return model
