
#include "metisfl/controller/scaling/participants_scaler.h"

namespace metisfl::controller
{

  absl::flat_hash_map<std::string, double>
  ParticipantsScaler::ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerDescriptor> &all_states,
      const absl::flat_hash_map<std::string, LearnerDescriptor *> &participating_states,
      const absl::flat_hash_map<std::string, TaskExecutionMetadata *> &participating_metadata)
  {

    /*
     * For a single (active or participating) learner the scaling factor is the identity value (=1).
     * For multiple learners, the scaling factors are the weighted average of all participants' identities (=1/N).
     */
    auto num_participants = participating_states.size();
    absl::flat_hash_map<std::string, double> scaling_factors;

    if (all_states.size() == 1)
    {
      // If running the federation with only one learner,
      // then its contribution value is always 1.
      auto learner_id = all_states.begin()->first;
      scaling_factors[learner_id] = 1;
    }
    else if (num_participants == 1)
    {
      // Else we run the federation with multiple learners.
      auto learner_id = participating_states.begin()->first;
      scaling_factors[learner_id] = 1;
    }
    else
    {

      for (const auto &[learner_id, state] : participating_states)
      {
        double scaling_factor =
            static_cast<double>(1) / static_cast<double>(num_participants);
        scaling_factors[learner_id] = scaling_factor;
      }
    }

    return scaling_factors;
  }

} // namespace metisfl::controller
