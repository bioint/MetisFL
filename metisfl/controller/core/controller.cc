
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {

// Constructor
Controller::Controller(const GlobalTrainParams &global_train_params,
                       const ModelStoreParams &model_store_params)
    : global_train_params_(move(global_train_params)) {
  model_manager_ =
      std::make_unique<ModelManager>(global_train_params, model_store_params);
  learner_manager_ = std::make_unique<LearnerManager>();

  scaler_ = CreateScaler(global_train_params.scaling_factor);
  scheduler_ = CreateScheduler(global_train_params_.communication_protocol);
  selector_ = CreateSelector();
}

// Public methods
absl::StatusOr<std::string> Controller::AddLearner(const Learner &learner) {
  return learner_manager_->AddLearner(learner);
}

absl::Status Controller::RemoveLearner(const std::string &learner_id) {
  RETURN_IF_ERROR(learner_manager_->ValidateLearnerId(learner_id));

  model_manager_->EraseModels(learner_id);

  learner_manager_->RemoveLearner(learner_id);

  return absl::OkStatus();
}

absl::Status Controller::StartTraining() {
  if (!model_manager_.->IsInitialized())
    return absl::FailedPreconditionError("Model is not initialized.");

  learner_manager_->ScheduleAll();

  return absl::OkStatus();
}

absl::Status Controller::TrainDone(const TrainDoneRequest &request) override {
  auto learner_id = request.learner_id();
  auto task_id = request.task_id();

  model_manager_.->InsertModel(learner_id, request.model());
  training_metadata_[task_id] = request.metadata();

  auto learner_ids = learner_manager_->GetLearnerIds();
  auto to_schedule = scheduler_->ScheduleNext(learner_id, learner_ids);

  if (!to_schedule.empty()) {
    // Doing the scheduling first so that we don't wait for the aggregation
    learner_manager_->ScheduleTasks(to_schedule);

    std::vector<std::string> selected_ids =
        selector_->Select(to_schedule, learner_ids);

    // FIXME:
    auto scaling_factors = scaler_->ComputeScalingFactors(selected_ids);

    model_manager_.->UpdateModel(selected_ids, scaling_factors);
  }

  return absl::OkStatus();
}

void Controller::Shutdown() { learner_manager_->Shutdown(); }

absl::flat_hash_map<std::string, double> Controller::ComputeScalingFactors(
    const std::vector<std::string> &selected_learners) {
  // TODO:
}

// FIXME:
void UpdateLearnersTaskTemplates(std::vector<std::string> &learners) {
  const auto &communication_protocol =
      global_train_params_.communication_protocol;
  if (communication_protocol == "SemiSynchronous" &&
      (global_iteration_ == 2 ||
       global_train_params_.semi_sync_recompute_num_updates)) {
    // Finds the slowest learner.
    float ms_per_epoch_slowest = std::numeric_limits<float>::min();
    for (const auto &learner_id : learners) {
      const auto &metadata = training_metadata_[learner_id].front();
      if (metadata.processing_ms_per_epoch() > ms_per_epoch_slowest) {
        ms_per_epoch_slowest = metadata.processing_ms_per_epoch();
      }
    }

    // Calculates the allowed time for training.
    float t_max = static_cast<float>(global_train_params_.semi_sync_lambda) *
                  ms_per_epoch_slowest;

    // Updates the task templates based on the slowest learner.
    for (const auto &learner_id : learners) {
      const auto &metadata = training_metadata_[learner_id].front();

      auto processing_ms_per_batch = metadata.processing_ms_per_batch();
      if (processing_ms_per_batch == 0) {
        PLOG(ERROR) << "Processing ms per batch is zero. Setting to 1.";
        processing_ms_per_batch = 1;
      }
      int num_local_updates = std::ceil(t_max / processing_ms_per_batch);

      auto &task_template = train_params_[learner_id];
      task_template.set_num_local_updates(num_local_updates);
    }
  }
}
}  // namespace metisfl::controller
