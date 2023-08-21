#ifndef METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_
#define METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_

#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {
class ModelManager {
 public:
  ModelManager(const GlobalTrainParams &global_train_params,
               const ModelStoreParams &model_store_params);

  void InitializeAggregator(const DType_Type tensor_dtype);

  const Model &GetModel() const { return model_.model(); }

  const TensorQuantifier &GetModelSize() const;

  void SetNumContributors(const int num_contributors) {
    model_.set_num_contributors(num_contributors);
  }

  void SetInitialModel(const Model &model);

  bool IsInitialized() const { return is_initialized_; }

  void InsertModel(std::string learner_id, Model model);

  void EraseModels(const std::vector<std::string> &learner_id,
                   const Model model);

 private:
  bool is_initialized_;
  FederatedModel model_;
  std::mutex model_store_mutex_;
  std::unique_ptr<AggregationFunction> aggregator_;
  GlobalTrainParams global_train_params_;
  std::unique_ptr<ScalingFunction> scaler_;
  std::unique_ptr<ModelStore> model_store_;
};
}  // namespace metisfl::controller

#endif  // METISFL_CONTROLLER_CORE_MODEL_MANAGER_H_