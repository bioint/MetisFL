
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {

// Constructor
Controller::Controller(const GlobalTrainParams &global_train_params,
                       const ModelStoreParams &model_store_params)
    : global_train_params_(move(global_train_params)) {
  model_manager_ =
      std::make_unique<ModelManager>(global_train_params, model_store_params);
  learner_manager_ = std::make_unique<LearnerManager>();

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

absl::Status Controller::SetInitialModel(const Model &model) {
  if (model_manager_->IsInitialized())
    return absl::FailedPreconditionError("Model is already initialized.");

  return model_manager_->SetInitialModel(model);
}

absl::Status Controller::StartTraining() {
  if (!model_manager_->IsInitialized())
    return absl::FailedPreconditionError("Model is not initialized.");

  learner_manager_->ScheduleAll();

  return absl::OkStatus();
}

absl::Status Controller::TrainDone(const TrainDoneRequest &request) override {
  auto learner_id = request.learner_id();
  auto task_id = request.task_id();

  model_manager_.->InsertModel(learner_id, request.model());
  learner_manager_->UpdateMetadata(task_id, request.metadata());

  auto learner_ids = learner_manager_->GetLearnerIds();
  auto to_schedule = scheduler_->ScheduleNext(learner_id, learner_ids);

  if (!to_schedule.empty()) {
    // Doing the scheduling first so that we don't wait for the aggregation
    learner_manager_->ScheduleTasks(to_schedule);

    std::vector<std::string> selected_ids =
        selector_->Select(to_schedule, learner_ids);

    // FIXME:
    auto scaling_factors = ComputeScalingFactors(selected_ids);

    model_manager_.->UpdateModel(selected_ids, scaling_factors);
  }

  return absl::OkStatus();
}

void Controller::Shutdown() { learner_manager_->Shutdown(); }

// Private methods
void Controller::ComputeScalingFactors(
    const std::vector<std::string> &selected_ids) {
  auto scaling_factor = global_train_params_.scaling_factor;

  if (scaling_factor == "NumCompletedBatches") {
    auto num_completed_batches =
        model_manager_.->GetNumCompletedBatches(selected_ids);
    return GetBatchesScalingFactors(num_completed_batches);
  } else if (scaling_factor == "NumParticipants") {
    auto num_learners = learner_manager_->GetNumLearners();
    return GetParticipantsScalingFactors(num_learners, selected_ids);
  } else if (scaling_factor == "NumTrainingExamples") {
    auto num_training_examples =
        model_manager_.->GetNumTrainingExamples(selected_ids);
    return GetDatasetScalingFactors(num_training_examples);
  } else {
    PLOG(FATAL) << "Unsupported scaling factor.";
  }
}
}  // namespace metisfl::controller
