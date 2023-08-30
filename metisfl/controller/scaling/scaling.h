#ifndef METISFL_METISFL_CONTROLLER_SCALING_H_
#define METISFL_METISFL_CONTROLLER_SCALING_H_

#include "absl/container/flat_hash_map.h"

namespace metisfl::controller {
class Scaling {
 public:
  static absl::flat_hash_map<std::string, double> GetDatasetScalingFactors(
      const absl::flat_hash_map<std::string, int> &num_training_examples) {
    absl::flat_hash_map<std::string, double> scaling_factors;

    long total_training_examples = 0;

    for (const auto &[_, num_examples] : num_training_examples) {
      total_training_examples += num_examples;
    }

    for (const auto &[learner_id, num_examples] : num_training_examples) {
      scaling_factors[learner_id] =
          static_cast<double>(num_examples) /
          static_cast<double>(total_training_examples);
    }

    return scaling_factors;
  }

  static absl::flat_hash_map<std::string, double> GetParticipantsScalingFactors(
      const std::vector<std::string> &selected_learner_ids) {
    int num_participants = selected_learner_ids.size();
    absl::flat_hash_map<std::string, double> scaling_factors;

    for (const auto &learner_id : selected_learner_ids) {
      scaling_factors[learner_id] =
          static_cast<double>(1) / static_cast<double>(num_participants);
    }

    return scaling_factors;
  };

  static absl::flat_hash_map<std::string, double> GetBatchesScalingFactors(
      const absl::flat_hash_map<std::string, int> &completed_batches) {
    absl::flat_hash_map<std::string, double> scaling_factors;

    long total_batches = 0;

    for (const auto &[_, num_batches] : completed_batches) {
      total_batches += num_batches;
    }

    for (const auto &[learner_id, num_batches] : completed_batches) {
      double scaling_factor =
          static_cast<double>(num_batches) / static_cast<double>(total_batches);
      scaling_factors[learner_id] = scaling_factor;
    }

    return scaling_factors;
  }
};
}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCALING_H_
