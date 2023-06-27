import numpy as np

from fhe import CKKS

if __name__ == "__main__":
    ckks = CKKS(4096, 52, "/metisfl/metisfl/resources/fheparams/cryptoparams")
    ckks.load_crypto_params()

    rng = np.random.default_rng()
    random_data = rng.random(5)

    number_of_learners = 2
    number_of_elems = 3
    learners_data = [[1 for _ in range(number_of_elems)]] * number_of_learners
    scaling_factors = [0.5] * number_of_learners
    print("Original learners data:", learners_data)
    print("Original scaling factors:", scaling_factors)

    learners_data_encrypted = []
    print("Encrypting...")
    for learner_data in learners_data:
        encrypted_res = ckks.encrypt(learner_data)
        learners_data_encrypted.append(encrypted_res)

    print("Decrypting...")
    for learner_data_enc in learners_data_encrypted:
        decrypted_res = ckks.decrypt(learner_data_enc, number_of_elems)
        print("Decrypted learner data: ", decrypted_res)

    print("Computing Private Weighted Average...")
    weighted_average_encrypted = ckks.compute_weighted_average(
        learners_data_encrypted, scaling_factors)
    print(ckks.decrypt(weighted_average_encrypted, number_of_elems))
