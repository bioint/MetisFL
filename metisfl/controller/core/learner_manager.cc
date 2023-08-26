#include "metisfl/controller/core/learner_manager.h"

namespace metisfl::controller {
LearnerManager::LearnerManager()
    : learners_(),
      learners_stub_(),
      train_params_(),
      eval_params_(),
      learners_mutex_(),
      scheduling_pool_(2),
      run_tasks_cq_(),
      eval_tasks_cq_() {
  std::thread run_tasks_digest_t_(&Controller::DigestTrainResponses, this);
  run_tasks_digest_t_.detach();

  std::thread eval_tasks_digest_t_(&Controller::DigestEvaluateResponses, this);
  eval_tasks_digest_t_.detach();
}

absl::StatusOr<std::string> AddLearner(const Learner &learner) override {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);

  const std::string learner_id =
      GenerateLearnerId(learner.hostname(), learner.port());

  if (learners_.contains(learner_id)) {
    return absl::AlreadyExistsError("Learner has already joined.");
  }

  learners_[learner_id] = learner;
  learners_stub_[learner_id] = CreateLearnerStub(learner_id);

  return learner_id;
}

absl::Status RemoveLearner(const std::string &learner_id) override {
  std::lock_guard<std::mutex> learners_guard(learners_mutex_);

  if (!learners_.contains(learner_id)) {
    return absl::NotFoundError("Learner does not exist.");
  }

  learners_.erase(learner_id);
  learners_stub_.erase(learner_id);
  train_params_.erase(learner_id);
  eval_params_.erase(learner_id);

  return absl::OkStatus();
}

void ScheduleAll() {
  scheduling_pool_.push_task([this, learner_id] { ScheduleTasks(learners_); });
}

void Schedule(const std::vector<std::string> &learner_ids) {
  scheduling_pool_.push_task(
      [this, learner_ids] { ScheduleTasks(learner_ids); });
}

void Shutdown() {
  run_tasks_cq_.Shutdown();
  eval_tasks_cq_.Shutdown();
  scheduling_pool_.wait_for_tasks();
  model_store_->Shutdown();
}

LearnerStub CreateLearnerStub(const std::string &learner_id) {
  auto hostname = learners_[learner_id].hostname();
  auto port = learners_[learner_id].port();
  auto target = absl::StrCat(hostname, ":", port);
  auto &root_certificate = learners_[learner_id].root_certificate_bytes();
  auto &public_certificate = learners_[learner_id].public_certificate_bytes();

  auto ssl_creds = grpc::InsecureChannelCredentials();

  if (!root_certificate.empty() && !public_certificate.empty()) {
    grpc::SslCredentialsOptions ssl_opts;
    ssl_opts.pem_root_certs = root_certificate;
    ssl_opts.pem_cert_chain = public_certificate;
    ssl_creds = grpc::SslCredentials(ssl_opts);
  }

  auto channel = grpc::CreateChannel(target, ssl_creds);
  return LearnerService::NewStub(channel);
}

absl::Status ValidateLearner(const std::string &learner_id) const {
  const auto &learner = learners_.find(learner_id);

  if (learner == learners_.end())
    return absl::NotFoundError("Learner does not exist.");

  return absl::OkStatus();
}

void ScheduleTasks(const std::vector<std::string> &learner_ids) {
  for (const auto &learner_id : learner_ids) {
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    SendTrainAsync(learner_id);

    // TODO: should we wait before sending the evaluation task?

    SendEvaluateAsync(learner_id);

    UpdateLearnersTaskTemplates(to_schedule);
  }
}

void SendEvaluateAsync(const std::string &learner_id) {
  auto task_id = metisfl::controller::GenerateRadnomId();
  EvaluationMetadataMap evaluation_metadata;
  evaluation_metadata_[task_id] = evaluation_metadata;

  EvaluateRequest request;
  *request.mutable_task_id() = task_id;
  *request.mutable_model() = model_manager_.GetModel();
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

void DigestEvaluateResponses() {
  void *got_tag;
  bool ok = false;

  auto &cq_ = eval_tasks_cq_;
  while (cq_.Next(&got_tag, &ok)) {
    auto *call = static_cast<AsyncLearnerEvalCall *>(got_tag);
    GPR_ASSERT(ok);

    if (call) {
      if (call->status.ok()) {
        (*metadata_.at(call->metadata_ref_idx)
              .mutable_eval_task_received_at())[call->learner_id] =
            TimeUtil::GetCurrentTime();

        (*community_evaluations_.at(call->comm_eval_ref_idx)
              .mutable_evaluations())[call->learner_id] = call->reply;
      } else {
        PLOG(ERROR) << "EvaluateModel RPC request to learner: "
                    << call->learner_id
                    << " failed with error: " << call->status.error_message();
      }
    }
    delete call;
  }
}

void SendTrainAsync(const std::string &learner_id) {
  TrainRequest request;
  *request.mutable_task_id() = metisfl::controller::GenerateRadnomId();
  *request.mutable_model() = model_manager_.GetModel();
  *request.mutable_params() = train_params_[learner_id];

  auto *call = new AsyncLearnerRunTaskCall;
  auto &cq = run_tasks_cq_;
  auto learner_stub = CreateLearnerStub(learner_id);

  call->learner_id = learner_id;
  call->response_reader =
      learner_stub->PrepareAsyncTrain(&call->context, request, &cq);
  call->response_reader->StartCall();
  call->response_reader->Finish(&call->reply, &call->status, (void *)call);
}

void DigestTrainResponses() {
  void *got_tag;
  bool ok = false;

  auto &cq_ = run_tasks_cq_;

  while (cq_.Next(&got_tag, &ok)) {
    auto *call = static_cast<AsyncLearnerRunTaskCall *>(got_tag);
    GPR_ASSERT(ok);

    if (call) {
      if (!call->status.ok()) {
        PLOG(ERROR) << "RunTask RPC request to learner: " << call->learner_id
                    << " failed with error: " << call->status.error_message();
      }
    }
    delete call;
  }
}

}  // namespace metisfl::controller