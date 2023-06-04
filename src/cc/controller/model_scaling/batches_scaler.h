
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_BATCHES_SCALER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_BATCHES_SCALER_H_

#include "src/cc/controller/model_scaling/scaling_function.h"
#include "src/proto/metis.pb.h"

namespace projectmetis::controller {

class BatchesScaler : public ScalingFunction {
 public:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState> &all_states,
      const absl::flat_hash_map<std::string, LearnerState*> &participating_states,
      const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &participating_metadata) override;

  inline std::string Name() override {
    return "BatchesScaler";
  }
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_BATCHES_SCALER_H_
