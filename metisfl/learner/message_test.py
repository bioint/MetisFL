import numpy as np

from metisfl.learner.message_helper import MessageHelper
from metisfl.encryption.homomorphic import HomomorphicEncryption
from metisfl.encryption.helper import generate_keys

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
print(weights, test)
assert np.allclose(test, weights[0])
