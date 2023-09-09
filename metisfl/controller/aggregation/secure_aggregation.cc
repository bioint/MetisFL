
#include "metisfl/controller/aggregation/secure_aggregation.h"

namespace metisfl::controller {

SecAgg::SecAgg(int batch_size, int scaling_factor_bits,
               std::string crypto_context) {
  ckks_.reset(new CKKS(batch_size, scaling_factor_bits, crypto_context));
}

Model SecAgg::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  std::vector<double> local_models_contrib_value;
  if (pairs.size() == 1) {
    local_models_contrib_value.emplace_back(1.0f);
  } else {
    for (const auto &pair : pairs) {
      const auto scale = pair.front().second;
      local_models_contrib_value.emplace_back(scale);
    }
  }
  Model global_model;
  const auto &sample_model = pairs.front().front().first;
  global_model.mutable_tensors()->CopyFrom(sample_model->tensors());

  auto total_tensors = global_model.tensors_size();
#pragma omp parallel for
  for (int var_idx = 0; var_idx < total_tensors; ++var_idx) {
    std::vector<std::string> local_tensor_ciphertexts;
    for (const auto &pair : pairs) {
      const auto *model = pair.front().first;
      local_tensor_ciphertexts.emplace_back(model->tensors(var_idx).value());
    }
    auto pwa_result =
        ckks_->Aggregate(local_tensor_ciphertexts, local_models_contrib_value);
    *global_model.mutable_tensors(var_idx)->mutable_value() = pwa_result;
  }
  return global_model;
}

void SecAgg::Reset() {}

}  // namespace metisfl::controller
