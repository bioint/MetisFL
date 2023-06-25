
#ifndef METISFL_METISFL_CONTROLLER_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
#define METISFL_METISFL_CONTROLLER_SCALING_TRAIN_DATASET_SIZE_SCALER_H_

#include "metisfl/controller/scaling/scaling_function.h"
#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {

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

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_SCALING_TRAIN_DATASET_SIZE_SCALER_H_
