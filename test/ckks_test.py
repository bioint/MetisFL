import numpy as np
import tempfile

from metisfl.encryption.fhe import CKKS
from metisfl.common.logger import MetisLogger


def test_ckks_api(
        batch_size,
        scaling_factor_bits,
        learners_data,
        scaling_factors,
        number_of_elems):

    crypto_context_path = tempfile.NamedTemporaryFile().name
    public_key_path = tempfile.NamedTemporaryFile().name
    private_key_path = tempfile.NamedTemporaryFile().name

    MetisLogger.info("Generating crypto context and keys...")
    ckks_scheme = CKKS.gen_crypto_params_files(batch_size, scaling_factor_bits,
                             crypto_context_path, public_key_path, private_key_path)


    learners_data_enc = []
    for learners_data in learners_data:
        MetisLogger.info("Encrypting...")
        ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
                        public_key_path, private_key_path)
        enc = ckks_scheme.encrypt(learners_data)
        learners_data_enc.append(enc)
        
        MetisLogger.info("Decrypting...")
        ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
                        public_key_path, private_key_path)
        learners_data_dec = ckks_scheme.decrypt(enc, number_of_elems)
        MetisLogger.info("Learners Data Decrypted: {}".format(learners_data_dec))


    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
                       public_key_path, private_key_path)

    pwa_enc = ckks_scheme.aggregate(learners_data_enc, scaling_factors)

    ckks_scheme = CKKS(batch_size, scaling_factor_bits, crypto_context_path,
                          public_key_path, private_key_path)
    pwa_dec = ckks_scheme.decrypt(pwa_enc, number_of_elems)
    MetisLogger.info("Aggregated (Decrypted) Result: {}".format(pwa_dec))


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

    # Define batch size and scaling factor bits of CKKS scheme.
    batch_size = 8192
    scaling_factor_bits = 40

    # Define demo example learners data and corresponding scaling factors.
    # Case 1: We examine if the number of elements are multiples of the batch size.
    number_of_learners = 2
    number_of_elems = 2 * batch_size
    learners_data = [[1 for _ in range(number_of_elems)]] * number_of_learners
    scaling_factors = [0.5] * number_of_learners
    # Just print the first 100 elements per learner.
    MetisLogger.info("Original learners data: {}".format(
        [x[:100] for x in learners_data]))
    MetisLogger.info("Original scaling factors: {}".format(
        np.array(scaling_factors[:number_of_learners])))
    test_ckks_api(batch_size, scaling_factor_bits, learners_data,
                  scaling_factors, number_of_elems)

    # Case 2: We examine if the number of elements are *NOT* multiples of the batch size.
    number_of_learners = 2
    number_of_elems = (2 * batch_size) + 1
    learners_data = [[2 for _ in range(number_of_elems)]] * number_of_learners
    scaling_factors = [0.5] * number_of_learners
    # Just print the first 100 elements per learner.
    MetisLogger.info("Original learners data: {}".format(
        [x[:100] for x in learners_data]))
    MetisLogger.info("Original scaling factors: {}".format(
        np.array(scaling_factors[:number_of_learners])))
    test_ckks_api(batch_size, scaling_factor_bits, learners_data,
                  scaling_factors, number_of_elems)
