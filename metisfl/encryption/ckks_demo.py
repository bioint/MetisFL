import os

from fhe import CKKS
from metisfl.utils.metis_logger import MetisLogger


def encrypt(ckks_scheme, data):
    MetisLogger.info("Encrypting...")
    ckks_scheme.load_crypto_context()
    ckks_scheme.load_public_key()

    learners_data_encrypted = []
    for x in data:
        res_enc = ckks_scheme.encrypt(x)
        learners_data_encrypted.append(res_enc)
    MetisLogger.info("Encrypting is complete.")

    return learners_data_encrypted


def decrypt(ckks_scheme, data_enc, number_of_elems):
    # Make sure the input data is a list.
    if not isinstance(data_enc, list):
        data_enc = [data_enc]
    MetisLogger.info("Decrypting...")
    ckks_scheme.load_crypto_context()
    ckks_scheme.load_private_key()

    data_dec = []
    for x_enc in data_enc:
        res_dec = ckks_scheme.decrypt(x_enc, number_of_elems)
        data_dec.append(res_dec)
    MetisLogger.info("Decrypting is complete.")

    return data_dec


def pwa(ckks_scheme, data_enc, scaling_factors):
    MetisLogger.info("Computing Private Weighted Average...")
    ckks_scheme.load_crypto_context()
    pwa_res = ckks_scheme.compute_weighted_average(
        data_enc, scaling_factors)
    MetisLogger.info("Private Weighted Average computation is complete.")
    return pwa_res


if __name__ == "__main__":

    """
    Through this demo we test the encryption, decryption and private weighted
    aggregation functions of the CKKS scheme. 
    
    To test each operation (encrypt, decrypt, pwa) in isolation and see which 
    crypto parameters are required to perform each operation, we create and 
    pass a separate CKKS scheme object at every function call. The corresponding 
    function loads the required crypto parameters through the given ckks scheme. 
    
    Specifically, for each operation we need the following crypto params: 
        encryption -> (crypto_context, public)
        decryption -> (crypto_context, private)
        pwa -> (crypto_context)     
    """

    # Define demo example learners data and corresponding scaling factors.
    number_of_learners = 2
    number_of_elems = 3
    learners_data = [[1 for _ in range(number_of_elems)]] * number_of_learners
    scaling_factors = [0.5] * number_of_learners
    MetisLogger.info("Original learners data: {}".format(learners_data))
    MetisLogger.info("Original scaling factors: {}".format(scaling_factors))

    crypto_params_dir = "/tmp/cryptoparams"
    if not os.path.exists(crypto_params_dir):
        os.makedirs(crypto_params_dir)

    # Define batch size and scaling factor bits of CKKS scheme.
    batch_size = 4096
    scaling_factor_bits = 52

    MetisLogger.info("Generating crypto context and keys...")
    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_params_dir)
    ckks_scheme.gen_crypto_context_and_keys()

    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_params_dir)
    learners_data_enc = encrypt(ckks_scheme, learners_data)

    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_params_dir)
    learners_data_dec = decrypt(ckks_scheme, learners_data_enc, number_of_elems)
    MetisLogger.info("Learners Data Decrypted: {}".format(learners_data_dec))

    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_params_dir)
    pwa_enc = pwa(ckks_scheme, learners_data_enc, scaling_factors)

    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_params_dir)
    pwa_dec = decrypt(ckks_scheme, pwa_enc, number_of_elems)
    MetisLogger.info("Aggregated (Decrypted) Result: {}".format(pwa_dec))
