
#include "projectmetis/controller/model_aggregation/private_weighted_average.h"

#include "projectmetis/proto/model.pb.h"

namespace projectmetis::controller {

FederatedModel
PWA::Aggregate(
    std::vector<std::pair<const Model*, double>>& pairs) {

  std::vector<float> scalingFactors;
  for (const auto &pair : pairs) {
    const auto scale = (float) pair.second;
    scalingFactors.emplace_back(scale);
  }

  // Initializes the community model.
  FederatedModel community_model;
  const auto& sample_model = pairs.front().first;
  // Loop over each variable one-by-one.
  for (int i = 0; i < sample_model->variables_size(); ++i) {
    auto* variable = community_model.mutable_model()->add_variables();
    variable->set_name(sample_model->variables(i).name());
    variable->set_trainable(sample_model->variables(i).trainable());
    *variable->mutable_ciphertext_tensor()->mutable_spec() =
        sample_model->variables(i).ciphertext_tensor().spec();

    std::vector<std::string> learners_Data;
    for (const auto &pair : pairs) {
      const auto *model = pair.first;
      learners_Data.emplace_back(
          model->variables(i).ciphertext_tensor().values());
    }
    // ComputeWeightedAverage assumes that each learner's contribution value,
    // scaling factor is already normalized / scaled.
    std::string pwa_result =
        fhe_helper_.computeWeightedAverage(learners_Data, scalingFactors);
    *variable->mutable_ciphertext_tensor()->mutable_values() = pwa_result;
  }

  // Sets the number of contributors to the number of input models.
  community_model.set_num_contributors(pairs.size());
  return community_model;
}

} // namespace projectmetis::controller
