#ifndef METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_
#define METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_

#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {
class ModelManager {
 public:
  ModelManager(const GlobalTrainParams &global_train_params,
               const ModelStoreParams &model_store_params);

  void InitializeAggregator(const DType_Type tensor_dtype);

  const Model &GetModel() const { return model_; }

  const TensorQuantifier &GetModelSize() const;

  void SetInitialModel(const Model &model);

  bool IsInitialized() const { return is_initialized_; }

  void InsertModel(std::string learner_id, Model model);

  void UpdateModel(std::string learner_id, Model model);

  void EraseModels(
      const std::string &task_id, const std::vector<std::string> &learner_ids,
      const absl::flat_hash_map<std::string, double> &scaling_factors);

 private:
  int GetStrideLength(const int num_learners) const;

  int GetLineageLength(std::string &learner_id) const;

  std::map<std::string, std::vector<const Model *>> SelectModels(
      const std::vector<std::pair<std::string, int>> &to_select_block);

  std::vector<std::vector<std::pair<Model *, double>>> Getstd::vector<std::vector<std::pair<Model *, double>>>(
      std::map<std::string, std::vector<const Model *>> selected_models,
      const absl::flat_hash_map<std::string, double> scaling_factors);

  vodi RecordTime();

  bool is_initialized_;
  Model model_;
  GlobalTrainParams global_train_params_;
  ModelMetadataMap metadata_;

  std::mutex model_store_mutex_;
  std::unique_ptr<AggregationFunction> aggregator_;
  std::unique_ptr<ModelStore> model_store_;
};
}  // namespace metisfl::controller

#endif  // METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_