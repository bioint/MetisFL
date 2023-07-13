from typing import List, Tuple
import tensorflow as tf


def get_model(
    metrics: List[str] = ["accuracy"],
    input_shape: Tuple[int] = (28, 28, 1),
    dense_units_per_layer: List[int] = [256, 128],
    num_classes: int = 10) -> tf.keras.Model:
    """A helper function to create a simple sequential model.

    Args:
        input_shape (tuple, optional): The input shape. Defaults to (28, 28).
        dense_units_per_layer (list, optional): Number of units per Dense layer. Defaults to [256, 128].
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
        model.add(Dense(units=units, activation="relu"))

    # Add the output layer
    model.add(Dense(num_classes, activation="softmax"))
    
    return model