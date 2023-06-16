
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_

#include "metisfl/controller/scaling/scaling_function.h"
#include "metisfl/proto/metis.pb.h"

namespace projectmetis::controller {

class TrainDatasetSizeScaler : public ScalingFunction {
 public:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState> &all_states,
      const absl::flat_hash_map<std::string, LearnerState*> &participating_states,
      const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &participating_metadata) override;

  inline std::string Name() override {
    return "TrainDatasetSizeScaler";
  }
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
