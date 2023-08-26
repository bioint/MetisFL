#ifndef METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_
#define METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_

#include "absl/container/flat_hash_map.h"
#include "metisfl/controller/core/types.h"

namespace metisfl::controller {
class LearnerManager {
 public:
  LearnerManager();
  ~LearnerManager() = default;

  // Getters/Setters
  TrainingMetadataMap GetTrainingMetadata() { return training_metadata_; }

  EvaluationMetadataMap GetEvaluationMetadata() { return evaluation_metadata_; }

  void UpdateMetadata(const std::string &task_id, const stf::string &learner_id,
                      const TrainingMetadata metadata);

  // Public methods
  absl::StatusOr<std::string> AddLearner(const Learner &learner);

  std::vector<std::string> GetLearnerIds() const;

  absl::Status RemoveLearner(const std::string &learner_id);

  void ScheduleAll();

  void Schedule(const std::vector<std::string> &learner_ids);

  absl::flat_hash_map<std::string, int> GetNumTrainingExamples(
      const std::vector<std::string> &learner_ids);

  absl::flat_hash_map<std::string, int> GetNumCompletedBatches(
      const std::vector<std::string> &learner_ids);

  void Shutdown();

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
  grpc::CompletionQueue run_tasks_cq_;
  grpc::CompletionQueue eval_tasks_cq_;

  // learner_id -> value
  LearnersMap learners_;
  LearnerStubMap learners_stub_;
  TrainParamsMap train_params_;
  EvaluationParamsMap eval_params_;

  // task_id -> learner_id
  TaskLearnerMap task_learner_map_;

  // task_id -> metadata
  TrainingMetadataMap training_metadata_;
  EvaluationMetadataMap evaluation_metadata_;

  // learner_id -> num_training_examples
  abs::flat_hash_map<std::string, int> num_training_examples_;

  // learner_id -> num_completed_batches in latest training task
  abs::flat_hash_map<std::string, double> num_completed_batches_;
};
}  // namespace metisfl::controller