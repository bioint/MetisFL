
#include "projectmetis/controller/model_scaling/dataset_size_scaler.h"

namespace projectmetis::controller {

absl::flat_hash_map<std::string,
                    double> DatasetSizeScaler::ComputeScalingFactors(
    const FederatedModel &community_model,
    const absl::flat_hash_map<std::string, LearnerState> &states) {
  long dataset_size = 0;
  for (const auto&[_, state] : states) {
    dataset_size += state.learner().dataset_spec().num_training_examples();
  }

  absl::flat_hash_map<std::string, double> scaling_factors;
  for (const auto&[learner_id, state] : states) {
    long
        num_examples = state.learner().dataset_spec().num_training_examples();
    double scaling_factor =
        static_cast<double>(num_examples) / static_cast<double>(dataset_size);
    scaling_factors[learner_id] = scaling_factor;
  }

  return scaling_factors;
}

} // namespace projectmetis::controller
