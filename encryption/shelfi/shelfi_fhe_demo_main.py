import numpy as np
import os
from shelfi_fhe_helper import ShelfiFheHelper

# Dummy test example to evaluate the above functions!
if __name__ == "__main__":
    fhe_helper = ShelfiFheHelper(encryption_scheme="ckks", batch_size=8192, scale_factor_bits=52,
                                 cryptoparams_dir="/metis/cryptoparams")
    iterations = range(0, 2)
    num_models = 2
    for i in iterations:
        print("Iteration: ", i)
        model1 = [np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
                  np.array([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]])]
        model2 = [np.array([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]),
                  np.array([[[13.0, 14.0], [15.0, 16.0]], [[17.0, 18.0], [19.0, 20.0]]])]

        models = [model1, model2]
        models_contribution_values = [1 / num_models] * num_models
        model = fhe_helper.encrypted_aggregation(models, models_contribution_values)
        print("PWA-model:")
        for matrix in model:
            print(matrix)
