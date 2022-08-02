
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_DATASET_SIZE_SCALER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_DATASET_SIZE_SCALER_H_

#include "projectmetis/controller/model_scaling/scaling_function.h"

namespace projectmetis::controller {

class DatasetSizeScaler : public ScalingFunction {
 public:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState> &states) override;

  inline std::string name() override {
    return "DI";
  }
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_DATASET_SIZE_SCALER_H_
