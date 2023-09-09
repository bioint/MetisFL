
import tempfile
from metisfl.encryption.fhe import CKKS

# get some temporary file paths
crypto_context_path = tempfile.NamedTemporaryFile().name
public_key_path = tempfile.NamedTemporaryFile().name
private_key_path = tempfile.NamedTemporaryFile().name

batch_size = 8192
scaling_factor_bits = 40

# generate the crypto context and keys
CKKS.gen_crypto_params_files(batch_size, scaling_factor_bits,
                             crypto_context_path, public_key_path, private_key_path)

# get an instance of the CKKS class
ckks = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
            public_key_path, private_key_path)

# encrypt some data
encrypted_data = ckks.encrypt([1, 2, 3, 4, 5])
# print(encrypted_data)

# decrypt the data
ckks = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
            public_key_path, private_key_path)
decrypted_data = ckks.decrypt(encrypted_data, 1)

# print the decrypted data
print(decrypted_data)
