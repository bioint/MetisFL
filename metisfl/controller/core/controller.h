
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
#include "metisfl/controller/scaling/scaling.h"

using google::protobuf::util::TimeUtil;

namespace metisfl::controller {
class Controller {
 public:
  Controller(const GlobalTrainParams &global_train_params,
             const ModelStoreParams &model_store_params);

  ~Controller() = default;

  uint32_t GetNumLearners() const { return learners_.size(); }

  TrainingMetadataMap GetTrainingMetadata() {
    return learner_manager_->GetTrainingMetadata();
  }

  EvaluationMetadataMap GetEvaluationMetadata() {
    return learner_manager_->GetEvaluationMetadata();
  }

  ModelMetadata GetModelMetadata() {
    return model_manager_->GetModelMetadata();
  }

  absl::StatusOr<std::string> AddLearner(const Learner &learner);

  absl::Status SetInitialModel(const Model &model);

  absl::Status RemoveLearner(const std::string &learner_id);

  absl::Status StartTraining();

  absl::Status TrainDone(const TrainDoneRequest &task);

  void Shutdown() = 0;

 private:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const std::vector<std::string> &selected_learners);

  GlobalTrainParams global_train_params_;

  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<LearnerManager> learner_manager_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Selector> selector_;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
