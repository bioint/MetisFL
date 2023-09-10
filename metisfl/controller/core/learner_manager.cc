#include "metisfl/controller/core/learner_manager.h"

namespace metisfl::controller {
using google::protobuf::util::TimeUtil;

// Constructor
LearnerManager::LearnerManager()
    : learners_(),
      train_params_(),
      eval_params_(),
      learners_mutex_(),
      scheduling_pool_(2),
      train_tasks_cq_(),
      eval_tasks_cq_() {
  std::thread run_tasks_digest_t_(&LearnerManager::DigestTrainResponses, this);
  std::thread eval_tasks_digest_t_(&LearnerManager::DigestEvaluateResponses,
                                   this);
  run_tasks_digest_t_.detach();
  eval_tasks_digest_t_.detach();
}

// Public methods
absl::StatusOr<std::string> LearnerManager::AddLearner(
    const Learner &learner, bool is_semi_sync,
    const std::string &scaling_factor) {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);

  std::string learner_id =
      GenerateLearnerId(learner.hostname(), learner.port());

  if (learners_.contains(learner_id)) {
    return absl::AlreadyExistsError("Learner has already joined.");
  }
  learners_[learner_id] = learner;

  TrainParams train_params;
  if (is_semi_sync) {
    train_params.add_metadata("processing_ms_per_batch");
    train_params.add_metadata("processing_ms_per_epoch");
  }
  if (scaling_factor == "NumTrainingExamples")
    train_params.add_metadata("num_training_examples");
  if (scaling_factor == "NumCompletedBatches")
    train_params.add_metadata("num_completed_batches");

  train_params_[learner_id] = train_params;

  return learner_id;
}

std::vector<std::string> LearnerManager::GetLearnerIds() const {
  std::vector<std::string> learner_ids;
  for (const auto &[key, learner] : learners_) {
    learner_ids.push_back(key);
  }
  return learner_ids;
}

absl::Status LearnerManager::RemoveLearner(const std::string &learner_id) {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);

  if (!learners_.contains(learner_id)) {
    return absl::NotFoundError("Learner does not exist.");
  }

  learners_.erase(learner_id);
  train_params_.erase(learner_id);
  eval_params_.erase(learner_id);

  return absl::OkStatus();
}

void LearnerManager::ScheduleTrain(const std::vector<std::string> &learner_ids,
                                   const Model &model) {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);
  for (const auto &learner_id : learner_ids) {
    scheduling_pool_.push_task(
        [this, learner_id, model] { SendTrainAsync(learner_id, model); });
  }
}

void LearnerManager::ScheduleEvaluate(
    const std::vector<std::string> &learner_ids, const Model &model) {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);
  for (const auto &learner_id : learner_ids) {
    scheduling_pool_.push_task(
        [this, learner_id, model] { SendEvaluateAsync(learner_id, model); });
  }
}

void LearnerManager::Shutdown() {
  train_tasks_cq_.Shutdown();
  eval_tasks_cq_.Shutdown();
  scheduling_pool_.wait_for_tasks();
}

void LearnerManager::UpdateTrainResults(const Task &task,
                                        const std::string &learner_id,
                                        const TrainResults &results) {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);
  auto task_id = task.id();
  train_results_[task_id] = results;
  latest_train_results_[learner_id] = results;
  *tasks_[task_id].mutable_received_at() = task.received_at();
  *tasks_[task_id].mutable_completed_at() = task.completed_at();
}

absl::flat_hash_map<std::string, int> LearnerManager::GetNumTrainingExamples(
    const std::vector<std::string> &learner_ids) {
  absl::flat_hash_map<std::string, int> num_training_examples;

  for (const auto &learner_id : learner_ids) {
    num_training_examples[learner_id] = (int)latest_train_results_[learner_id]
                                            .metadata()
                                            .find("num_training_examples")
                                            ->second;
  }
  return num_training_examples;
}

absl::flat_hash_map<std::string, int> LearnerManager::GetNumCompletedBatches(
    const std::vector<std::string> &learner_ids) {
  absl::flat_hash_map<std::string, int> num_completed_batches;

  for (const auto &learner_id : learner_ids) {
    num_completed_batches[learner_id] = (int)latest_train_results_[learner_id]
                                            .metadata()
                                            .find("num_completed_batches")
                                            ->second;
  }

  return num_completed_batches;
}

void LearnerManager::UpdateTrainParams(
    const std::vector<std::string> &learner_ids, const int semi_sync_lambda) {
  // Finds the slowest learner.
  float ms_per_epoch_slowest = std::numeric_limits<float>::min();
  for (const auto &learner_id : learner_ids) {
    auto processing_ms_per_epoch = train_results_[learner_id]
                                       .metadata()
                                       .find("processing_ms_per_epoch")
                                       ->second;
    if (processing_ms_per_epoch > ms_per_epoch_slowest) {
      ms_per_epoch_slowest = processing_ms_per_epoch;
    }
  }

  // Calculates the allowed time for training.
  float t_max = static_cast<float>(semi_sync_lambda) * ms_per_epoch_slowest;

  // Updates the task templates based on the slowest learner.
  for (const auto &learner_id : learner_ids) {
    auto processing_ms_per_batch = train_results_[learner_id]
                                       .metadata()
                                       .find("processing_ms_per_batch")
                                       ->second;
    if (processing_ms_per_batch == 0) {
      // FIXME: Better error handling.
      LOG(ERROR) << "Processing ms per batch is zero. Setting to 1.";
      processing_ms_per_batch = 1;
    }
    int num_local_updates = std::ceil(t_max / processing_ms_per_batch);

    train_params_[learner_id].set_num_local_updates(num_local_updates);
  }
}

// Private methods
LearnerStub LearnerManager::CreateLearnerStub(const std::string &learner_id) {
  auto hostname = learners_[learner_id].hostname();
  auto port = learners_[learner_id].port();
  auto target = absl::StrCat(hostname, ":", port);
  auto &root_certificate = learners_[learner_id].root_certificate_bytes();

  auto ssl_creds = grpc::InsecureChannelCredentials();

  if (!root_certificate.empty()) {
    grpc::SslCredentialsOptions ssl_opts;
    ssl_opts.pem_root_certs = root_certificate;
    ssl_creds = grpc::SslCredentials(ssl_opts);
  }

  auto channel = grpc::CreateChannel(target, ssl_creds);
  return LearnerService::NewStub(channel);
}

bool LearnerManager::ValidateLearner(const std::string &learner_id) const {
  return learners_.contains(learner_id);
}

void LearnerManager::SendTrainAsync(const std::string &learner_id,
                                    const Model &model) {
  auto task_id = GenerateRadnomId();
  tasks_[task_id] = Task();
  *tasks_[task_id].mutable_id() = task_id;
  *tasks_[task_id].mutable_learner_id() = learner_id;
  *tasks_[task_id].mutable_sent_at() = TimeUtil::GetCurrentTime();

  TrainRequest request;
  *request.mutable_task() = tasks_[task_id];
  *request.mutable_model() = model;
  *request.mutable_params() = train_params_[learner_id];

  auto *call = new AsyncLearnerRunTaskCall;
  auto &cq = train_tasks_cq_;
  auto learner_stub = CreateLearnerStub(learner_id);

  call->learner_id = learner_id;
  call->response_reader =
      learner_stub->PrepareAsyncTrain(&call->context, request, &cq);
  call->response_reader->StartCall();
  call->response_reader->Finish(&call->reply, &call->status, (void *)call);
}

void LearnerManager::DigestTrainResponses() {
  void *got_tag;
  bool ok = false;
  auto &cq_ = train_tasks_cq_;
  while (cq_.Next(&got_tag, &ok)) {
    auto *call = static_cast<AsyncLearnerRunTaskCall *>(got_tag);
    GPR_ASSERT(ok);

    if (call) {
      if (!call->status.ok()) {
        LOG(ERROR) << "Train RPC request to learner: " << call->learner_id
                   << " failed with error: " << call->status.error_message();
      }
    }
    delete call;
  }
}

void LearnerManager::SendEvaluateAsync(const std::string &learner_id,
                                       const Model &model) {
  auto task_id = GenerateRadnomId();
  tasks_[task_id] = Task();
  *tasks_[task_id].mutable_id() = task_id;
  *tasks_[task_id].mutable_learner_id() = learner_id;
  *tasks_[task_id].mutable_sent_at() = TimeUtil::GetCurrentTime();

  EvaluateRequest request;
  *request.mutable_task() = tasks_[task_id];
  *request.mutable_model() = model;
  *request.mutable_params() = eval_params_[learner_id];

  auto *call = new AsyncLearnerEvalCall;
  auto &cq = eval_tasks_cq_;
  auto learner_stub = CreateLearnerStub(learner_id);

  call->learner_id = learner_id;
  call->response_reader =
      learner_stub->PrepareAsyncEvaluate(&call->context, request, &cq);
  call->response_reader->StartCall();
  call->response_reader->Finish(&call->reply, &call->status, (void *)call);
}

void LearnerManager::DigestEvaluateResponses() {
  void *got_tag;
  bool ok = false;
  auto &cq_ = eval_tasks_cq_;
  while (cq_.Next(&got_tag, &ok)) {
    auto *call = static_cast<AsyncLearnerEvalCall *>(got_tag);
    GPR_ASSERT(ok);

    if (call) {
      if (call->status.ok()) {
        const std::string &task_id = call->reply.task().id();
        evaluation_results_[task_id] = call->reply.results();
        *tasks_[task_id].mutable_received_at() =
            call->reply.task().received_at();
        *tasks_[task_id].mutable_completed_at() =
            call->reply.task().completed_at();
      } else {
        LOG(ERROR) << "EvaluateModel RPC request to learner: "
                   << call->learner_id
                   << " failed with error: " << call->status.error_message();
      }
    }
    delete call;
  }
}

}  // namespace metisfl::controller