import os

from fhe import CKKS
from metisfl.utils.metis_logger import MetisLogger

if __name__ == "__main__":
    crypto_params_dir = "/tmp/cryptoparams"
    if not os.path.exists(crypto_params_dir):
        os.makedirs(crypto_params_dir)
    ckks = CKKS(4096, 52, crypto_params_dir)

    MetisLogger.info("Generating crypto context and keys...")
    ckks.gen_crypto_context_and_keys()
    MetisLogger.info("Loading params...")
    ckks.load_crypto_params()

    number_of_learners = 2
    number_of_elems = 3
    learners_data = [[1 for _ in range(number_of_elems)]] * number_of_learners
    scaling_factors = [0.5] * number_of_learners
    MetisLogger.info("Original learners data: {}".format(learners_data))
    MetisLogger.info("Original scaling factors: {}".format(scaling_factors))

    learners_data_encrypted = []
    MetisLogger.info("Encrypting...")
    for learner_data in learners_data:
        encrypted_res = ckks.encrypt(learner_data)
        learners_data_encrypted.append(encrypted_res)

    MetisLogger.info("Decrypting...")
    for learner_data_enc in learners_data_encrypted:
        decrypted_res = ckks.decrypt(learner_data_enc, number_of_elems)
        MetisLogger.info("Decrypted learner data: {}".format(decrypted_res))

    MetisLogger.info("Computing Private Weighted Average...")
    weighted_average_encrypted = ckks.compute_weighted_average(
        learners_data_encrypted, scaling_factors)
    MetisLogger.info(ckks.decrypt(weighted_average_encrypted, number_of_elems))
