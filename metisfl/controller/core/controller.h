
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_

#include <glog/logging.h>

#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/learner_manager.h"
#include "metisfl/controller/core/model_manager.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/controller/scaling/scaling.h"

using google::protobuf::util::TimeUtil;

namespace metisfl::controller {
class Controller {
  GlobalTrainParams global_train_params_;

  std::unique_ptr<ModelManager> model_manager_;
  std::unique_ptr<LearnerManager> learner_manager_;
  std::unique_ptr<Scheduler> scheduler_;
  std::unique_ptr<Selector> selector_;

 public:
  Controller(const GlobalTrainParams &global_train_params,
             const ModelStoreParams &model_store_params);

  ~Controller() = default;

  // Getters
  int GetGlobalIteration() { return scheduler_->GetGlobalIteration(); }

  TaskLearnerMap GetTasksMap() { return learner_manager_->GetTasksMap(); }

  TrainResultsMap GetTrainResults() {
    return learner_manager_->GetTrainResults();
  }

  EvaluationResultsMap GetEvaluationResults() {
    return learner_manager_->GetEvaluationResults();
  }

  ModelMetadataMap GetModelMetadata() {
    return model_manager_->GetModelMetadata();
  }

  std::string GetLearnerId(std::string task_id) {
    return learner_manager_->GetLearnerId(task_id);
  }

  // Public methods
  absl::StatusOr<std::string> AddLearner(const Learner &learner);

  absl::Status SetInitialModel(const Model &model);

  absl::Status RemoveLearner(std::string learner_id);

  absl::Status StartTraining();

  absl::Status TrainDone(const TrainDoneRequest &task);

  void Shutdown();

 private:
  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const std::vector<std::string> &selected_learners);
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
