import numpy as np

from fhe import CKKS

if __name__ == "__main__":
    ckks = CKKS(4096, 52, "/metisfl/metisfl/resources/fheparams/cryptoparams")
    ckks.load_crypto_params()

    rng = np.random.default_rng()
    random_data = rng.random(5)
    random_data_encrypted = ckks.encrypt(random_data)
    print(random_data_encrypted)


