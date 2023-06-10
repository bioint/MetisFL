
#include <omp.h>

#include "metisfl/controller/model_aggregation/private_weighted_average.h"
#include "metisfl/proto/model.pb.h"

namespace projectmetis::controller {

PWA::PWA(const HEScheme &he_scheme) {
  he_scheme_ = he_scheme;
  if (he_scheme_.has_fhe_scheme()) {
    fhe_helper_ = FHE_Helper(
        he_scheme_.name(),
        he_scheme_.fhe_scheme().batch_size(),
        he_scheme_.fhe_scheme().scaling_bits());
      fhe_helper_.load_crypto_params();
  } else {
      throw std::runtime_error("Unsupported homomorphic encryption scheme.");
  }
}

FederatedModel
PWA::Aggregate(std::vector<std::vector<std::pair<const Model*, double>>>& pairs) {

  // Throughout this implementation, we use the first model provided in the pair.
  // If only one learner is given for the aggregation step, then we set its scaling
  // factor value to 1, else we use the precomputed scaling factors.
  std::vector<float> scalingFactors;
  if (pairs.size() == 1) {
    scalingFactors.emplace_back(1);
  } else {
    for (const auto &pair : pairs) {
      const auto scale = (float) pair.front().second;
      scalingFactors.emplace_back(scale);
    }
  }

  // Initializes the community model.
  FederatedModel community_model;
  const auto& sample_model = pairs.front().front().first;
  // Loop over each variable one-by-one.
  #pragma omp parallel for
  for (int i = 0; i < sample_model->variables_size(); ++i) {
    auto* variable = community_model.mutable_model()->add_variables();
    variable->set_name(sample_model->variables(i).name());
    variable->set_trainable(sample_model->variables(i).trainable());
    *variable->mutable_ciphertext_tensor()->mutable_tensor_spec() =
        sample_model->variables(i).ciphertext_tensor().tensor_spec();

    std::vector<std::string> learners_data;
    for (const auto &pair : pairs) {
      const auto *model = pair.front().first;
      learners_data.emplace_back(
          model->variables(i).ciphertext_tensor().tensor_spec().value());
    }
    // ComputeWeightedAverage assumes that each learner's contribution value,
    // scaling factor is already normalized / scaled.
    std::string pwa_result =
        fhe_helper_.computeWeightedAverage(learners_data, scalingFactors);
    *variable->mutable_ciphertext_tensor()->mutable_tensor_spec()->mutable_value() = pwa_result;
  }

  // Sets the number of contributors to the number of input models.
  community_model.set_num_contributors(pairs.size());
  return community_model;
}

void PWA::Reset() {
  // pass
}

} // namespace projectmetis::controller
