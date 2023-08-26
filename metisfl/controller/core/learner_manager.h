#ifndef METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_
#define METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_

#include "metisfl/controller/core/types.h"

namespace metisfl::controller {
class LearnerManager {
 public:
  LearnerManager();
  ~LearnerManager() = default;

  absl::StatusOr<std::string> AddLearner(const Learner &learner);

  std::vector<std::string> GetLearnerIds() const;

  absl::Status RemoveLearner(const std::string &learner_id);

  void ScheduleAll();

  void Schedule(const std::vector<std::string> &learner_ids);

 private:
  LearnerStub CreateLearnerStub(const std::string &learner_id);

  absl::StatusOr<std::string> ValidateLearner(const std::string &learner_id);

  void ScheduleTasks(const std::vector<std::string> &learner_ids);

  void SendEvaluateAsync(const std::string &learner_id);

  void DigestEvaluateResponses();

  void SendTrainAsync(const std::string &learner_id);

  void DigestTrainResponses();

  std::mutex learners_mutex_;
  BS::thread_pool scheduling_pool_;

  LearnersMap learners_;
  LearnerStubMap learners_stub_;
  TrainParamsMap train_params_;
  EvaluationParamsMap eval_params_;

  TaskLearnerMap task_learner_map_;
  TrainingMetadataMap training_metadata_;
  EvaluationMetadataMap evaluation_metadata_;

  grpc::CompletionQueue run_tasks_cq_;
  grpc::CompletionQueue eval_tasks_cq_;
};
}  // namespace metisfl::controller