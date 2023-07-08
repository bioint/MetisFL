import tensorflow as tf

from metisfl.models.keras.optimizers.fed_prox import FedProx


def get_model(metrics=["accuracy"], optimizer_name="MomentumSGD"):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
                                     activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3),
                                     activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    if optimizer_name.lower() in ["vanillasgd", "momentumsgd"]:
        optimizer = tf.keras.optimizers.SGD()
    elif optimizer_name.lower() == "fedprox":
        optimizer = FedProx()
    else:
        raise RuntimeError("Not a supported optimizer.")
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=metrics)
    return model
