import os
from typing import List, Tuple
import tensorflow as tf
import numpy as np

from metisfl.learner.message import MessageHelper
from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.helpers.ckks import generate_keys

batch_size = 8192
scaling_factor_bits = 40
cc = "/tmp/cc.txt"
pk = "/tmp/pk.txt"
prk = "/tmp/prk.txt"

generate_keys(batch_size, scaling_factor_bits, cc, pk, prk)

scheme = HomomorphicEncryption(batch_size, scaling_factor_bits, cc, pk, prk)

test = np.random.rand(2, 2)

# No encryption
helper = MessageHelper()
model = helper.weights_to_model_proto([test])
weights = helper.model_proto_to_weights(model)
assert np.allclose(test, weights[0])

# Encryption
helper = MessageHelper(scheme)
model = helper.weights_to_model_proto([test])
weights = helper.model_proto_to_weights(model)

assert np.allclose(test, weights[0])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


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


weights = get_model(
    input_shape=(2, 2, 1),
    dense_units_per_layer=[2, 2],
).get_weights()

# No encryption
helper = MessageHelper()
model = helper.weights_to_model_proto(weights)
weights_out = helper.model_proto_to_weights(model)
for w1, w2 in zip(weights, weights_out):
    assert np.allclose(w1, w2)


# Encryption
helper = MessageHelper(scheme)
model = helper.weights_to_model_proto(weights)
weights_out = helper.model_proto_to_weights(model)
for w1, w2 in zip(weights, weights_out):
    assert np.allclose(w1, w2, atol=1e-3)
