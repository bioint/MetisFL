from metisfl.pybind.fhe import fhe
import numpy as np

if __name__ == "__main__":
    batchsize, scalingfactorbits = 4096, 52
    ckks = fhe.CKKS(4096, 52, "resources/fheparams/cryptoparams/")


    # Generate the necessary CryptoParameters
    ckks.gen_crypto_context_and_key_gen()
    ckks.load_crypto_params()

    data_dimesions = 100000
    scalingFactors = [0.5, 0.2, 0.3]
    learner1_data_layer_1 = np.random.rand(data_dimesions)
    learner2_data_layer_1 = np.random.rand(data_dimesions)
    learner3_data_layer_1 = np.random.rand(data_dimesions)

    print("Learner1: ", learner1_data_layer_1)
    print("Learner2: ", learner2_data_layer_1)
    print("Learner3: ", learner3_data_layer_1)

    # encrypting
    print("Encrypting...")
    enc_res_learner_1 = ckks.encrypt(learner1_data_layer_1, 1)
    enc_res_learner_2 = ckks.encrypt(learner2_data_layer_1, 1)
    enc_res_learner_3 = ckks.encrypt(learner3_data_layer_1, 1)
    print("Encryption done")

    # decrypting
    print("Decrypting...")
    print("Learner1 (decrypted): ", ckks.decrypt(enc_res_learner_1, data_dimesions, 1))
    print("Learner2 (decrypted): ", ckks.decrypt(enc_res_learner_2, data_dimesions, 1))
    print("Learner3 (decrypted): ", ckks.decrypt(enc_res_learner_3, data_dimesions, 1))
    print("Decryption done")

    # learner1_data_actual = []
    # learner2_data_actual = []
    # learner3_data_actual = []
    # for i in range(data_dimesions):
    #     learner1_data_actual.append(learner1_data_layer_1[i])
    #     learner2_data_actual.append(learner2_data_layer_1[i])
    #     learner3_data_actual.append(learner3_data_layer_1[i])
    # '''dec_res1 = FHE_helper.decrypt(enc_res_learner_1, data_dimesions)
    # dec_res2 = FHE_helper.decrypt(enc_res_learner_2, data_dimesions)
    # dec_res3 = FHE_helper.decrypt(enc_res_learner_3, data_dimesions)
    # print(dec_res1)
    # print(dec_res2)
    # print(dec_res3)'''
    # three_learners_enc_data = [enc_res_learner_1, enc_res_learner_2, enc_res_learner_3]
    # # weighted average
    # PWA_res = ckks.compute_weighted_average(three_learners_enc_data, scalingFactors, data_dimesions)
    # print("pwa done")
    # # decryption required information about dimension of each layer of model
    # # decryption
    # dec_res = ckks.decrypt(PWA_res, data_dimesions, 1)
    # print("decrypt done")
    # print("result: 0.5*L1 + 0.2*L2 + 0.3*L3")
    # result = []
    # learner1_data_actual = [element * scalingFactors[0] for element in learner1_data_actual]
    # learner2_data_actual = [element * scalingFactors[1] for element in learner2_data_actual]
    # learner3_data_actual = [element * scalingFactors[2] for element in learner3_data_actual]
    # for i in range(len(learner1_data_actual)):
    #     result.append(learner1_data_actual[i] + learner2_data_actual[i] + learner3_data_actual[i])
    # # printing result
    # j = 0
    # for i in dec_res:
    #     print("computed: " + str(i) + " " + "actual: " + str(result[j]))
    #     j = j + 1
