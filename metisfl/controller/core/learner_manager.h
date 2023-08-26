#ifndef METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_
#define METISFL_CONTROLLER_CORE_LEARNER_MANAGER_H_

#include "metisfl/controller/core/types.h"

namespace metisfl::controller {
class LearnerManager {
 public:
  LearnerManager();
  ~LearnerManager() = default;

 private:
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