
#include "projectmetis/controller/model_scaling/batches_scaler.h"

namespace projectmetis::controller {

absl::flat_hash_map<std::string, double>
BatchesScaler::ComputeScalingFactors(
    const FederatedModel &community_model,
    const absl::flat_hash_map<std::string, LearnerState*> &states,
    const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &metadata) {

  /*
   * For a single learner the scaling factor is its completed batches value.
   * For multiple learners, the scaling factors are the weighted average of all completed batches.
   */
  absl::flat_hash_map<std::string, double> scaling_factors;
  if (metadata.size() == 1) {

    auto learner_id = metadata.begin()->first;
    auto num_batches =
        static_cast<double>(metadata.begin()->second->completed_batches());
    scaling_factors[learner_id] = num_batches;

  } else {

    long total_batches = 0;
    for (const auto&[_, meta] : metadata) {
      total_batches += (*meta).completed_batches();
    }

    for (const auto&[learner_id, meta] : metadata) {
      long num_batches = (*meta).completed_batches();
      double scaling_factor =
          static_cast<double>(num_batches) / static_cast<double>(total_batches);
      scaling_factors[learner_id] = scaling_factor;
    }

  }

  return scaling_factors;

}

} // namespace projectmetis::controller
