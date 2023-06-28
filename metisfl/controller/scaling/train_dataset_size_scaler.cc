
#include "metisfl/controller/scaling/train_dataset_size_scaler.h"

namespace metisfl::controller {

absl::flat_hash_map<std::string, double>
TrainDatasetSizeScaler::ComputeScalingFactors(
    const FederatedModel &community_model,
    const absl::flat_hash_map<std::string, LearnerState> &all_states,
    const absl::flat_hash_map<std::string, LearnerState*> &participating_states,
    const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &participating_metadata) {

  /*
   * For a single learner the scaling factor is equal to its raw dataset size value.
   * For multiple learners, the scaling factors are the weighted average of all raw values.
   */
  absl::flat_hash_map<std::string, double> scaling_factors;

  if (all_states.size() == 1) {
    // If running the federation with only one learner,
    // then its contribution value is always 1.
    auto learner_id = all_states.begin()->first;
    scaling_factors[learner_id] = 1;

  } else if (participating_states.size() == 1) {

    auto learner_id = participating_states.begin()->first;
    auto num_training_examples = static_cast<double>(
        participating_states.begin()->second->learner().dataset_spec().num_training_examples());
    scaling_factors[learner_id] = num_training_examples;

  } else {

    long total_training_examples = 0;
    for (const auto &[_, state]: participating_states) {
      total_training_examples += (*state).learner().dataset_spec().num_training_examples();
    }
    for (const auto &[learner_id, state]: participating_states) {
      long num_examples =
          (*state).learner().dataset_spec().num_training_examples();
      double scaling_factor =
          static_cast<double>(num_examples) / static_cast<double>(total_training_examples);
      scaling_factors[learner_id] = scaling_factor;
    }

  }

  return scaling_factors;

}

} // namespace metisfl::controller
