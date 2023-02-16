
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_

#include "projectmetis/controller/model_scaling/scaling_function.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

class TrainDatasetSizeScaler : public ScalingFunction {
 public:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState*> &states,
      const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &metadata) override;

  inline std::string Name() override {
    return "TrainDatasetSizeScaler";
  }
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
