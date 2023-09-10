#ifndef METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_
#define METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_

#include <google/protobuf/util/time_util.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/impl/codegen/async_unary_call.h>

#include <thread>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/core/types.h"

namespace metisfl::controller {
class LearnerManager {
  std::mutex learners_mutex_;
  BS::thread_pool scheduling_pool_;
  grpc::CompletionQueue train_tasks_cq_;
  grpc::CompletionQueue eval_tasks_cq_;

  // learner_id -> value
  // TODO: how does the controller gets these values?
  LearnersMap learners_;
  TrainParamsMap train_params_;
  EvaluationParamsMap eval_params_;

  // task_id -> {}
  TaskMap tasks_;
  TrainResultsMap train_results_;
  EvaluationResultsMap evaluation_results_;

  // learner_id -> {}
  TrainResultsMap latest_train_results_;

 public:
  LearnerManager();

  ~LearnerManager() = default;

  // Getters/Setters

  TaskMap GetTaskMap() { return tasks_; }
  TrainResultsMap GetTrainResults() { return train_results_; }
  EvaluationResultsMap GetEvaluationResults() { return evaluation_results_; }

  std::string GetLearnerId(const std::string &task_id) {
    return tasks_[task_id].learner_id();
  }

  void UpdateTrainResults(const Task &task, const std::string &learner_id,
                          const TrainResults &metadata);

  void UpdateTrainParams(const std::vector<std::string> &learner_ids,
                         const int semi_sync_lambda);

  // Public methods
  absl::StatusOr<std::string> AddLearner(const Learner &learner,
                                         const bool is_semi_sync,
                                         const std::string &scaling_factor);

  std::vector<std::string> GetLearnerIds() const;

  absl::Status RemoveLearner(const std::string &learner_id);

  bool ValidateLearner(const std::string &learner_id) const;

  void ScheduleTrain(const std::vector<std::string> &learner_ids,
                     const Model &model);

  void ScheduleEvaluate(const std::vector<std::string> &learner_ids,
                        const Model &model);

  absl::flat_hash_map<std::string, int> GetNumTrainingExamples(
      const std::vector<std::string> &learner_ids);

  absl::flat_hash_map<std::string, int> GetNumCompletedBatches(
      const std::vector<std::string> &learner_ids);

  void Shutdown();

 private:
  LearnerStub CreateLearnerStub(const std::string &learner_id);

  void SendEvaluateAsync(const std::string &learner_id, const Model &model);

  void DigestEvaluateResponses();

  void SendTrainAsync(const std::string &learner_id, const Model &model);

  void DigestTrainResponses();
};
}  // namespace metisfl::controller

#endif  // METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_