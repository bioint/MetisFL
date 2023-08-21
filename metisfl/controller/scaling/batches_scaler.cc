
#include "metisfl/controller/scaling/batches_scaler.h"

namespace metisfl::controller
{

  absl::flat_hash_map<std::string, double>
  BatchesScaler::ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerDescriptor> &all_states,
      const absl::flat_hash_map<std::string, LearnerDescriptor *> &participating_states,
      const absl::flat_hash_map<std::string, TrainingMetadata *> &participating_metadata)
  {

    /*
     * For a single (active or participating) learner the scaling factor is the identity value (=1).
     * For multiple learners, the scaling factors are the weighted average of all the participants' completed batches.
     */
    absl::flat_hash_map<std::string, double> scaling_factors;

    if (all_states.size() == 1)
    {
      // If running the federation with only one learner,
      // then its contribution value is always 1.
      auto learner_id = all_states.begin()->first;
      scaling_factors[learner_id] = 1;
    }
    else if (participating_metadata.size() == 1)
    {

      auto learner_id = participating_metadata.begin()->first;
      scaling_factors[learner_id] = 1;
    }
    else
    {

      long total_batches = 0;
      for (const auto &[_, meta] : participating_metadata)
      {
        total_batches += (*meta).completed_batches();
      }

      for (const auto &[learner_id, meta] : participating_metadata)
      {
        long num_batches = (*meta).completed_batches();
        double scaling_factor =
            static_cast<double>(num_batches) / static_cast<double>(total_batches);
        scaling_factors[learner_id] = scaling_factor;
      }
    }

    return scaling_factors;
  }

} // namespace metisfl::controller
