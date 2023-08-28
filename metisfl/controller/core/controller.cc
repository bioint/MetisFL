
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {

// Constructor
Controller::Controller(const GlobalTrainParams &global_train_params,
                       const ModelStoreParams &model_store_params) {
  global_train_params_ = global_train_params;

  model_manager_ =
      absl::make_unique<ModelManager>(global_train_params_, model_store_params);
  learner_manager_ = absl::make_unique<LearnerManager>();
  scheduler_ = CreateScheduler(global_train_params_.communication_protocol);
  selector_ = CreateSelector();
}

// Public methods
absl::StatusOr<std::string> Controller::AddLearner(const Learner &learner) {
  return learner_manager_->AddLearner(learner);
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

  learner_manager_->ScheduleAll(model_manager_->GetModel());

  return absl::OkStatus();
}

absl::Status Controller::TrainDone(const TrainDoneRequest &request) {
  auto learner_id = request.learner_id();
  auto task_id = request.task_id();

  model_manager_->InsertModel(learner_id, request.model());
  learner_manager_->UpdateMetadata(task_id, learner_id, request.metadata());

  auto learner_ids = learner_manager_->GetLearnerIds();
  auto to_schedule = scheduler_->ScheduleNext(learner_id, learner_ids.size());

  if (!to_schedule.empty()) {
    // Doing the scheduling first so that we don't wait for the aggregation
    learner_manager_->Schedule(to_schedule, model_manager_->GetModel());

    std::vector<std::string> selected_ids =
        selector_->Select(to_schedule, learner_ids);

    auto scaling_factors = ComputeScalingFactors(selected_ids);

    model_manager_->UpdateModel(selected_ids, scaling_factors);
  }

  return absl::OkStatus();
}

void Controller::Shutdown() {
  learner_manager_->Shutdown();
  model_manager_->Shutdown();
}

// Private methods
absl::flat_hash_map<std::string, double> Controller::ComputeScalingFactors(
    const std::vector<std::string> &selected_ids) {
  auto scaling_factor = global_train_params_.scaling_factor;

  if (scaling_factor == "NumCompletedBatches") {
    auto num_completed_batches =
        learner_manager_->GetNumCompletedBatches(selected_ids);
    return Scaling::GetBatchesScalingFactors(num_completed_batches);
  } else if (scaling_factor == "NumParticipants") {
    return Scaling::GetParticipantsScalingFactors(selected_ids);
  } else if (scaling_factor == "NumTrainingExamples") {
    auto num_training_examples =
        learner_manager_->GetNumTrainingExamples(selected_ids);
    return Scaling::GetDatasetScalingFactors(num_training_examples);
  } else {
    PLOG(FATAL) << "Unsupported scaling factor.";
  }
}
}  // namespace metisfl::controller
