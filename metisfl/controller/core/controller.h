
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_

#include <glog/logging.h>
#include <google/protobuf/util/time_util.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/impl/codegen/async_unary_call.h>

#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/model_manager.h"
#include "metisfl/controller/core/types.h"

namespace metisfl::controller {

class Controller {
 public:
  Controller(const GlobalTrainParams &global_train_params,
             const ModelStoreParams &model_store_params);

  ~Controller() = default;

  uint32_t GetNumLearners() const { return learners_.size(); }

  TrainingMetadataMap &GetTrainingMetadata() { return training_metadata_; }

  EvaluationMetadataMap &GetEvaluationMetadata() {
    return evaluation_metadata_;
  }

  ModelMetadata GetModelMetadata() {
    return model_manager_->GetModelMetadata();
  }

  std::vector<std::string> GetLearnerIds() const = 0;

  absl::Status SetInitialModel(const Model &model) = 0;

  absl::StatusOr<std::string> AddLearner(const Learner &learner) = 0;

  absl::Status StartTraining() = 0;

  absl::Status RemoveLearner(const std::string &learner_id) = 0;

  absl::Status TrainDone(const TrainDoneRequest &task) = 0;

  void Shutdown() = 0;

 private:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const std::vector<std::string> &selected_learners) = 0;

  GlobalTrainParams global_train_params_;

  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<LearnerManager> learner_manager_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Selector> selector_;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
