import SHELFI_FHE as m

import numpy as np
import gc


class ShelfiFheHelper(object):

    def __init__(self):
        self.FHE_helper = m.FHE_Helper("ckks", 8192, 52)
        self.FHE_helper.load_cyrpto_params()

    def encrypt_compute_decrypt(self, models, models_scaling_factors, total_model_parameters):
        encoded_models = list()
        for model in models:
            encoded_model = self.FHE_helper.encrypt(model)
            encoded_models.append(encoded_model)
            # Delete encoded model object; this is an FHE library instance.
            del encoded_model

        pwa_res = self.FHE_helper.computeWeightedAverage(encoded_models, models_scaling_factors)
        dec_res = self.FHE_helper.decrypt(pwa_res, total_model_parameters)

        # Delete the privately weighted aggregation model object; this is an FHE library instance.
        del pwa_res

        # Clear the list of encoded models.
        encoded_models.clear()

        return dec_res

    def encrypted_aggregation(self, learners_models, learners_weighting_values):
        norm_factor = sum(learners_weighting_values)
        learners_weighting_values = [float(val / norm_factor) for val in learners_weighting_values]

        base_model = learners_models[0]
        model_matrices_dtype = [matrix.dtype for matrix in base_model]
        model_matrices_shapes = [matrix.shape for matrix in base_model]
        model_matrices_cardinality = [matrix.size for matrix in base_model]
        total_model_parameters = sum(model_matrices_cardinality)

        dec_agg_res = self.encrypt_compute_decrypt(learners_models, learners_weighting_values, total_model_parameters)

        # Convert it to python list - due to subscribing
        aggregated_model = dec_agg_res
        tmp_final_model = []
        parameter_offset = 0
        for matrix_cardinality in model_matrices_cardinality:
            tmp_final_model.append(aggregated_model[parameter_offset: parameter_offset + matrix_cardinality])
            parameter_offset += matrix_cardinality

        # Construct the final model by bringing each array back to its required dimension and data type.
        final_model = []
        for midx, aggregated_matrix in enumerate(tmp_final_model):
            final_matrix = [float(val) for val in aggregated_matrix]
            final_matrix = np.array(final_matrix).astype(model_matrices_dtype[midx])
            final_matrix = final_matrix.reshape(model_matrices_shapes[midx])
            final_model.append(final_matrix)

        # Delete the decrypted aggregation object; this is an FHE library instance.
        del dec_agg_res
        # Clear any obsolete objects!
        gc.collect()

        return final_model
