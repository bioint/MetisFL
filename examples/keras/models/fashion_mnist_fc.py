import tensorflow as tf

from metisfl.models.keras.wrapper import MetisKerasModel



def get_model():
    """Prepare a simple dense model."""
    Dense = tf.keras.layers.Dense
    Flatten = tf.keras.layers.Flatten

    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(128, activation="relu", kernel_initializer="glorot_uniform"))
    model.add(Dense(10, activation="softmax", kernel_initializer="glorot_uniform"))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
