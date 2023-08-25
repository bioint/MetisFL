
#include "metisfl/controller/aggregation/secure_aggregation.h"

namespace metisfl::controller {

SecAgg::SecAgg(const int batch_size, const int scaling_factor_bits,
               const std::string &crypto_context) {
  encryption_scheme_.reset(new CKKS(batch_size, scaling_factor_bits));
  encryption_scheme_->LoadCryptoContextFromFile(crypto_context);
}

FederatedModel SecAgg::Aggregate(std::vector<std::vector<std::pair<Model *, double>>> &pairs) {
  // Throughout this implementation, we use the first model provided in the
  // pair. If only one learner is given for the aggregation step, then we set
  // its scaling factor value to 1, else we use the precomputed scaling factors.
  // We create scaling factors / local models contribution values as floats
  // because the signature of the computeWeightedAverage() function in the
  // FHE_Helper API is expecting a vector of floats.
  std::vector<float> local_models_contrib_value;
  if (pairs.size() == 1) {
    local_models_contrib_value.emplace_back(1.0f);
  } else {
    for (const auto &pair : pairs) {
      const auto scale = (float)pair.front().second;
      local_models_contrib_value.emplace_back(scale);
    }
  }
  FederatedModel global_model;
  const auto &sample_model = pairs.front().front().first;
  global_model.mutable_model()->mutable_tensors()->CopyFrom(
      sample_model->tensors());

  auto total_tensors = global_model.model().tensors_size();
  // Parallelize encrypted aggregation of model tensors.
#pragma omp parallel for
  for (int var_idx = 0; var_idx < total_tensors; ++var_idx) {
    std::vector<std::string> local_tensor_ciphertexts;
    for (const auto &pair : pairs) {
      const auto *model = pair.front().first;
      local_tensor_ciphertexts.emplace_back(model->tensors(var_idx).value());
    }
    // The `Aggregate` function assumes that each learner's contribution value,
    // scaling factor is already normalized / scaled.
    auto pwa_result = encryption_scheme_->Aggregate(local_tensor_ciphertexts,
                                                    local_models_contrib_value);
    *global_model.mutable_model()->mutable_tensors(var_idx)->mutable_value() =
        pwa_result;
  }

  // Sets the number of contributors to the number of input models.
  global_model.set_num_contributors(pairs.size());
  return global_model;
}

void SecAgg::Reset() {}

}  // namespace metisfl::controller
