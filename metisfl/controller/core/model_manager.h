#ifndef METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_
#define METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_

#include "absl/status/statusor.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/learner_manager.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/controller/scaling/scaling.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {
class ModelManager {
  bool is_initialized_ = false;
  Model model_;
  GlobalTrainParams global_train_params_;
  ModelMetadataMap metadata_;

  std::unique_ptr<AggregationFunction> aggregator_;
  std::unique_ptr<ModelStore> model_store_;

  LearnerManager *learner_manager_;
  Selector *selector_;

 public:
  ModelManager(LearnerManager *learner_manager, Selector *selector,
               const GlobalTrainParams &global_train_params,
               const ModelStoreParams &model_store_params);

  ~ModelManager() = default;

  // Getters
  Model GetModel() const { return model_; }

  ModelMetadataMap GetModelMetadata() const { return metadata_; }

  bool IsInitialized() const { return is_initialized_; }

  // Public methods
  absl::Status SetInitialModel(const Model &model);

  void InsertModel(std::string &learner_id, const Model &model);

  void UpdateModel(std::vector<std::string> &to_schedule,
                   std::vector<std::string> &learner_ids);

  void EraseModels(std::vector<std::string> &learner_ids);

  void Shutdown();

 private:
  std::string InitializeMetadata();

  int GetStrideLength(int num_learners) const;

  int GetLineageLength(std::string &learner_id) const;

  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const std::vector<std::string> &selected_ids) const;

  std::map<std::string, std::vector<const Model *>> SelectModels(
      std::string &update_id,
      std::vector<std::pair<std::string, int>> &to_select_block);

  std::vector<std::vector<std::pair<const Model *, double>>>
  GetAggregationPairs(
      std::map<std::string, std::vector<const Model *>> &selected_models,
      absl::flat_hash_map<std::string, double> &scaling_factors) const;

  void Aggregate(std::string &update_id,
                 std::vector<std::vector<std::pair<const Model *, double>>>
                     &to_aggregate_block);

  void RecordBlockSize(std::string &update_id, int block_size);

  void RecordAggregationTime(
      std::string &update_id,
      std::chrono::time_point<std::chrono::system_clock> &start);
};
}  // namespace metisfl::controller

#endif  // METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_