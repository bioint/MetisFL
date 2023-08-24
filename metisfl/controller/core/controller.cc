
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {
namespace {
using google::protobuf::util::TimeUtil;

class ControllerDefaultImpl : public Controller {
 public:
  ControllerDefaultImpl(const GlobalTrainParams &global_train_params,
                        const ModelStoreParams &model_store_params)
      : global_train_params_(move(global_train_params)),
        learners_(),
        learners_stub_(),
        learners_train_params_(),
        learners_mutex_(),
        scheduling_pool_(2),
        run_tasks_cq_(),
        eval_tasks_cq_() {
    model_manager_ =
        std::make_unique<ModelManager>(global_train_params, model_store_params);
    scaler_ = CreateScaler(global_train_params.scaling_factor);

    std::thread run_tasks_digest_t_(
        &ControllerDefaultImpl::DigestTrainResponses, this);
    run_tasks_digest_t_.detach();

    std::thread eval_tasks_digest_t_(
        &ControllerDefaultImpl::DigestEvaluateResponses, this);
    eval_tasks_digest_t_.detach();
  }

  uint32_t GetNumLearners() const override { return learners_.size(); }

  std::vector<std::string> GetLearnerIds() const override {
    std::vector<std::string> learner_ids;
    for (const auto &[key, learner] : learners_) {
      learner_ids.push_back(key);
    }
    return learner_ids;
  }

  virtual absl::StatusOr<std::string> AddLearner(
      const Learner &learner) override {
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    const std::string learner_id =
        GenerateLearnerId(learner.hostname(), learner.port());

    if (learners_.contains(learner_id)) {
      PLOG(INFO) << "Learner " << learner_id << " re-joined Federation.";
      return absl::AlreadyExistsError("Learner has already joined.");
    }

    learners_[learner_id] = learner;
    learners_stub_[learner_id] = CreateLearnerStub(learner_id);

    return learner_id;
  }

  absl::Status RemoveLearner(const std::string &learner_id) override {
    RETURN_IF_ERROR(ValidateLearner(learner_id));

    model_manager_->EraseModels(learner_id);

    std::lock_guard<std::mutex> learners_guard(learners_mutex_);
    auto it = learners_.find(learner_id);
    learners_.erase(it);
    learners_stub_.erase(learner_id);
    learners_train_params_.erase(learner_id);

    PLOG(INFO) << "Removed learner with id: " << learner_id << ".";

    return absl::OkStatus();
  }

  absl::Status StartTraining() override {
    if (!model_manager_.->IsInitialized())
      return absl::FailedPreconditionError("Model is not initialized.");

    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    // Schedule initial tasks for all learners.
    scheduling_pool_.push_task(
        [this, learner_id] { ScheduleTasks(learners_); });

    model_manager_->SetNumContributors(learners_.size());

    return absl::OkStatus();
  }

  absl::Status TrainDone(const TrainDoneRequest &request) override {
    auto learner_id = request.learner_id();
    auto task_id = request.task_id();

    model_manager_.->InsertModel(learner_id, request.model());
    training_metadata_[task_id] = request.metadata();

    auto to_schedule = scheduler_->ScheduleNext(learner_id, GetLearnerIds());
    if (!to_schedule.empty()) {
      // Doing the scheduling first so that we don't wait for the aggregation
      scheduling_pool_.push_task(
          [this, to_schedule] { ScheduleTasks(to_schedule); });

      std::vector<std::string> selected_ids =
          selector_->Select(to_schedule, GetLearnerIds());

      auto scaling_factors = scaler_->ComputeScalingFactors(selected_ids);

      model_manager_.->UpdateModel(task_id, selected_ids);
    }

    return absl::OkStatus();
  }

  TrainingMetadataMap GetTrainingMetadata() override {
    return training_metadata_;
  }

  EvaluationMetadataMap GetEvaluationMetadata() override {
    return evaluation_metadata_;
  }

  void Shutdown() override {
    run_tasks_cq_.Shutdown();
    eval_tasks_cq_.Shutdown();
    scheduling_pool_.wait_for_tasks();
    model_store_->Shutdown();
  }

 private:
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

  absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const std::vector<std::string> &selected_learners) override {
    // TODO:
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

  void UpdateLearnersTaskTemplates(std::vector<std::string> &learners) {
    // If we are running in a semi-synchronous setting, we need to update the
    // task templates based on the execution times of the previous global
    // iteration.
    const auto &communication_protocol =
        global_train_params_.communication_protocol;
    // We check if it is the 2nd global_iteration_, because the 1st
    // global_iteration_ refers to the very first initially scheduled task.
    // For the SemiSynchronous protocol, recomputation takes place only
    // if we are just starting the 2nd federation round or we want to
    // recompute the number of updates at every federation round
    // (i.e., dynamic assignment of local training steps).
    if (communication_protocol == "SemiSynchronous" &&
        (global_iteration_ == 2 ||
         global_train_params_.semi_sync_recompute_num_updates)) {
      // Finds the slowest learner.
      // float ms_per_batch_slowest = std::numeric_limits<float>::min();
      float ms_per_epoch_slowest = std::numeric_limits<float>::min();
      for (const auto &learner_id : learners) {
        const auto &metadata = training_metadata_[learner_id].front();
        // if (metadata.processing_ms_per_batch() > ms_per_batch_slowest) {
        //   ms_per_batch_slowest = metadata.processing_ms_per_batch();
        // }
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

        auto &task_template = learners_train_params_[learner_id];
        task_template.set_num_local_updates(num_local_updates);
      }

    }  // end-if.
  }

  void SendEvaluateAsync(const std::string &learner_id) {
    auto task_id = GenerateTaskId();
    EvaluationMetadataMap evaluation_metadata;
    evaluation_metadata_[task_id] = evaluation_metadata;

    EvaluateRequest request;
    *request.mutable_task_id() = task_id;
    *request.mutable_model() = model_manager_.GetModel();

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
    // This function loops indefinitely over the `eval_task_q` completion queue.
    // It stops when it receives a shutdown signal. Specifically, `cq._Next()`
    // will not return false until `cq_.Shutdown()` is called and all pending
    // tags have been digested.
    void *got_tag;
    bool ok = false;

    auto &cq_ = eval_tasks_cq_;
    // Block until the next result is available in the completion queue "cq".
    while (cq_.Next(&got_tag, &ok)) {
      // The tag is the memory location of the call object.
      auto *call = static_cast<AsyncLearnerEvalCall *>(got_tag);

      // Verify that the request was completed successfully. Note that "ok"
      // corresponds solely to the request for updates introduced by Finish().
      GPR_ASSERT(ok);

      if (call) {
        // If either a failed or successful response is received
        // then handle the content of the received response.
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
      }  // end if call

      delete call;

    }  // end of loop
  }

  void SendTrainAsync(const std::string &learner_id) {
    TrainRequest request;
    *request.mutable_task_id() = GenerateTaskId();
    *request.mutable_model() = model_manager_.GetModel();
    *request.mutable_params() = learners_train_params_[learner_id];

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
    // This function loops indefinitely over the `run_task_q` completion queue.
    // It stops when it receives a shutdown signal. Specifically, `cq._Next()`
    // will not return false until `cq_.Shutdown()` is called and all pending
    // tags have been digested.
    void *got_tag;
    bool ok = false;

    auto &cq_ = run_tasks_cq_;
    // Block until the next result is available in the completion queue "cq".
    while (cq_.Next(&got_tag, &ok)) {
      // The tag is the memory location of the call object.
      auto *call = static_cast<AsyncLearnerRunTaskCall *>(got_tag);

      // Verify that the request was completed successfully. Note that "ok"
      // corresponds solely to the request for updates introduced by Finish().
      GPR_ASSERT(ok);

      if (call) {
        // If either a failed or successful response is received
        // then handle the content of the received response.
        if (!call->status.ok()) {
          PLOG(ERROR) << "RunTask RPC request to learner: " << call->learner_id
                      << " failed with error: " << call->status.error_message();
        }
      }  // end if call

      delete call;

    }  // end of loop
  }

  std::unique_ptr<Scheduler> scheduler_ =
      CreateScheduler(global_train_params_.communication_protocol);
  std::unique_ptr<Selector> selector_ = CreateSelector();

  template <typename T>
  struct AsyncLearnerCall {
    std::string learner_id;
    T reply;
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<T>> response_reader;
  };

  // Handles responses to Train
  struct AsyncLearnerRunTaskCall : AsyncLearnerCall<Ack> {};

  // Handles responses to Evaluate
  struct AsyncLearnerEvalCall : AsyncLearnerCall<EvaluateResponse> {
    // Index to the community/global model evaluation metrics vector.
    uint32_t comm_eval_ref_idx;
    // Index to the metadata collection vector.
    uint32_t metadata_ref_idx;
    AsyncLearnerEvalCall() {
      comm_eval_ref_idx = 0;
      metadata_ref_idx = 0;
    }
  };
};

}  // namespace

std::unique_ptr<Controller> Controller::New(
    const GlobalTrainParams &global_train_params,
    const ModelStoreParams &model_store_params) {
  return absl::make_unique<ControllerDefaultImpl>(global_train_params,
                                                  model_store_params);
}

}  // namespace metisfl::controller
