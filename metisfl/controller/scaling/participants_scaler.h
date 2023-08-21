
#ifndef METISFL_METISFL_CONTROLLER_SCALING_PARTICIPANTS_SCALER_H_
#define METISFL_METISFL_CONTROLLER_SCALING_PARTICIPANTS_SCALER_H_

#include "metisfl/controller/scaling/scaling_function.h"

namespace metisfl::controller
{

  class ParticipantsScaler : public ScalingFunction
  {
  public:
    absl::flat_hash_map<std::string, double> ComputeScalingFactors(
        const FederatedModel &community_model,
        const absl::flat_hash_map<std::string, LearnerDescriptor> &all_states,
        const absl::flat_hash_map<std::string, LearnerDescriptor *> &participating_states,
        const absl::flat_hash_map<std::string, TrainingMetadata *> &participating_metadata) override;

    inline std::string Name() override
    {
      return "ParticipantsScaler";
    }
  };

} // namespace metisfl::controller

#endif // METISFL_METISFL_CONTROLLER_SCALING_PARTICIPANTS_SCALER_H_
