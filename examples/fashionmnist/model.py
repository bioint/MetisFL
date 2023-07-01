import tensorflow as tf


def get_model(
    input_shape: tuple = (28, 28),
    dense_units_per_layer: list = [128, 128],
    num_classes: int = 10,
) -> tf.keras.Model:
    """A helper function to create a simple sequential model.

    Args:
        input_shape (tuple, optional): The input shape. Defaults to (28, 28).
        dense_units_per_layer (list, optional): Number of units per Dense layer. Defaults to [128, 128].
        num_classes (int, optional): Shape of the output. Defaults to 10.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """

    # For convenience and readability
    Dense = tf.keras.layers.Dense
    Flatten = tf.keras.layers.Flatten

    # Create a sequential model
    model = tf.keras.models.Sequential()

    # Add the input layer
    model.add(Flatten(input_shape=input_shape))

    # Add the dense layers
    for units in dense_units_per_layer:
        model.add(Dense(units=units, activation="relu",
                  kernel_initializer="glorot_uniform"))

    # Add the output layer
    model.add(Dense(num_classes, activation="softmax",
              kernel_initializer="glorot_uniform"))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model