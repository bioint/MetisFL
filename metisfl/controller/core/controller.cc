
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {

// Constructor
Controller::Controller(const GlobalTrainParams &global_train_params,
                       const ModelStoreParams &model_store_params) {
  global_train_params_ = global_train_params;

  learner_manager_ = absl::make_unique<LearnerManager>();
  scheduler_ = CreateScheduler(global_train_params_.scheduler);
  selector_ = CreateSelector();
  model_manager_ =
      absl::make_unique<ModelManager>(learner_manager_.get(), selector_.get(),
                                      global_train_params_, model_store_params);
}

// Public methods
absl::StatusOr<std::string> Controller::AddLearner(const Learner &learner) {
  auto is_semi_sync =
      global_train_params_.scheduler == "SemiSynchronous";
  return learner_manager_->AddLearner(learner, is_semi_sync,
                                      global_train_params_.scaling_factor);
}

absl::Status Controller::RemoveLearner(std::string learner_id) {
  if (!learner_manager_->ValidateLearner(learner_id))
    return absl::NotFoundError("Learner does not exist.");

  std::vector<std::string> learner_ids = {learner_id};
  model_manager_->EraseModels(learner_ids);

  return learner_manager_->RemoveLearner(learner_id);
}

absl::Status Controller::SetInitialModel(const Model &model) {
  return model_manager_->SetInitialModel(model);
}

absl::Status Controller::StartTraining() {
  if (!model_manager_->IsInitialized())
    return absl::FailedPreconditionError("Model is not initialized.");

  auto learner_ids = learner_manager_->GetLearnerIds();
  std::vector<std::string> to_schedule;
  for (const auto &learner_id : learner_ids) {
    to_schedule = scheduler_->ScheduleNext(learner_id, learner_ids.size());
    if (!to_schedule.empty())
      learner_manager_->ScheduleTrain(to_schedule, model_manager_->GetModel());
  }

  return absl::OkStatus();
}

absl::Status Controller::TrainDone(const TrainDoneRequest &request) {
  std::lock_guard<std::mutex> model_manager_guard(model_manager_mutex_);
  std::lock_guard<std::mutex> learner_manager_guard(learner_manager_mutex_);

  auto task = request.task();
  auto learner_id = learner_manager_->GetLearnerId(task.id());
  learner_manager_->UpdateTrainResults(task, learner_id, request.results());
  model_manager_->InsertModel(learner_id, request.model());
  learner_manager_->ScheduleEvaluate({learner_id}, model_manager_->GetModel());

  auto learner_ids = learner_manager_->GetLearnerIds();
  auto to_schedule = scheduler_->ScheduleNext(learner_id, learner_ids.size());

  if (!to_schedule.empty()) {
    model_manager_->UpdateModel(to_schedule, learner_ids);
    learner_manager_->ScheduleTrain(to_schedule, model_manager_->GetModel());
    UpdateTrainParams(to_schedule);
  }

  return absl::OkStatus();
}

void Controller::UpdateTrainParams(
    const std::vector<std::string> &learner_ids) {
  if (global_train_params_.scheduler == "SemiSynchronous") {
    auto global_iteration = scheduler_->GetGlobalIteration();
    if (global_iteration == 2 ||
        global_train_params_.semi_sync_recompute_num_updates) {
      auto semi_sync_lambda = global_train_params_.semi_sync_lambda;
      learner_manager_->UpdateTrainParams(learner_ids, semi_sync_lambda);
    }
  }
}

void Controller::Shutdown() {
  learner_manager_->Shutdown();
  model_manager_->Shutdown();
}
}  // namespace metisfl::controller
