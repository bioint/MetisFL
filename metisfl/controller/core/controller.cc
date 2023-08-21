
#include "metisfl/controller/core/controller.h"

namespace metisfl::controller {
namespace {
using google::protobuf::util::TimeUtil;

class ControllerDefaultImpl : public Controller {
 public:
  ControllerDefaultImpl(const ServerParams &server_params,
                        const GlobalTrainParams &global_train_params,
                        const ModelStoreParams &model_store_params)
      : server_params_(move(server_params)),
        global_train_params_(move(global_train_params)),
        learners_(),
        learners_stub_(),
        learners_train_params_(),
        learners_mutex_(),
        scheduling_pool_(2),
        run_tasks_cq_(),
        eval_tasks_cq_() {
    model_manager_ =
        std::make_unique<ModelManager>(global_train_params, model_store_params);
    // One thread to handle learners' responses to Train requests.
    std::thread run_tasks_digest_t_(
        &ControllerDefaultImpl::DigestTrainResponses, this);
    run_tasks_digest_t_.detach();

    // One thread to handle learners' responses to Evaluate requests.
    std::thread eval_tasks_digest_t_(
        &ControllerDefaultImpl::DigestEvaluateResponses, this);
    eval_tasks_digest_t_.detach();
  }

  const ServerParams &GetServerParams() const override {
    return server_params_;
  }

  std::vector<std::string> GetLearnerIds() const override {
    std::vector<std::string> learner_ids;
    for (const auto &[key, learner] : learners_) {
      learner_ids.push_back(key);
    }
    return learner_ids;
  }

  uint32_t GetNumLearners() const override { return learners_.size(); }

  virtual absl::StatusOr<std::string> AddLearner(
      const LearnerDescriptor &learner) override {
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

    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    auto it = learners_.find(learner_id);
    if (it != learners_.end()) {
      PLOG(INFO) << "Removing learner from controller: " << learner_id;

      model_manager_->EraseModels(learner_id);
      learners_.erase(it);
      learners_stub_.erase(learner_id);
      learners_train_params_.erase(learner_id);

      return absl::OkStatus();
    } else
      return absl::NotFoundError("Learner is not part of the federation.");
  }

  absl::Status StartTraining() override {
    if (!model_manager_.->IsInitialized())
      return absl::FailedPreconditionError("Model is is not initialized.");

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

    *metadata_[task_id].add_completed_by_learner_id() = learner_id;
    (*metadata_[task_id].mutable_train_task_completed_at())[learner_id] =
        TimeUtil::GetCurrentTime();

    // This is a blocking call.
    model_manager_.->InsertModel(learner_id, request.model());

    if (!local_tasks_metadata_.contains(learner_id)) {
      local_tasks_metadata_[learner_id] = std::list<TrainingMetadata>();
    }
    local_tasks_metadata_[learner_id].push_back(task.metadata());

    auto to_schedule = scheduler_->ScheduleNext(learner_id, GetLearnerIds());

    std::vector<std::string> selected_ids =
        selector_->Select(to_schedule, GetLearnerIds());

    LearnersMap selected_learners;
    TaskMetadataMap selected_task_metadata;
    for (const auto &id : selected_ids) {
      if (learners_.contains(id) && local_tasks_metadata_.contains(id)) {
        selected_learners[id] = &learners_.at(id);
        selected_task_metadata[id] = &local_tasks_metadata_.at(id).back();
      }
    }

    model_manager_.->UpdateModel(selected_learners, selected_task_metadata);

    scheduling_pool_.push_task(
        [this, to_schedule] { ScheduleTasks(to_schedule); });

    return absl::OkStatus();
  }

  std::vector<RuntimeMetadata> GetRuntimeMetadataLineage(
      uint32_t num_steps) override {
    if (metadata_.empty()) {
      return {};
    }
    const auto &lineage = metadata_;
    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<RuntimeMetadata> lineage_head;
    auto iter = lineage.begin();
    while (lineage_head.size() < num_steps && iter != lineage.end()) {
      lineage_head.push_back(*iter);
      ++iter;
    }

    return lineage_head;
  }

  std::vector<CommunityModelEvaluation> GetEvaluationLineage(
      uint32_t num_steps) override {
    if (community_evaluations_.empty()) {
      return {};
    }

    const auto &lineage = community_evaluations_;

    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<CommunityModelEvaluation> lineage_head;
    auto iter = lineage.begin();
    while (lineage_head.size() < num_steps && iter != lineage.end()) {
      lineage_head.push_back(*iter);
      ++iter;
    }

    return lineage_head;
  }

  std::vector<TrainingMetadata> GetLocalTaskLineage(
      const std::string &learner_id, uint32_t num_steps) override {
    if (!local_tasks_metadata_.contains(learner_id)) {
      return {};
    }

    const auto &lineage = local_tasks_metadata_[learner_id];

    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<TrainingMetadata> lineage_head;
    auto iter = lineage.begin();
    while (lineage_head.size() < num_steps && iter != lineage.end()) {
      lineage_head.push_back(*iter);
      ++iter;
    }

    return lineage_head;
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

  void ScheduleTasks(const std::vector<std::string> &learner_ids) {
    for (const auto &learner_id : learner_ids) {
      std::lock_guard<std::mutex> learners_guard(learners_mutex_);
      std::stirng train_task_id = GenerateTaskId();

      metadata_.[task_id] = RuntimeMetadata();
      *metadata_.[task_id].mutable_started_at() = TimeUtil::GetCurrentTime();
      *metadata_.[task_id].add_assigned_to_learner_id() = learner_id;
      (*metadata_.[task_id].mutable_train_task_submitted_at())[learner_id] =
          TimeUtil::GetCurrentTime();

      SendTrainAsync(learner_id, task_id);

      // TODO: should we wait before sending the evaluation task?
      CommunityModelEvaluation community_eval;
      community_model_evaluations_[task_id] = community_eval;
      (*metadata_.[task_id].mutable_eval_task_submitted_at())[learner_id] =
          TimeUtil::GetCurrentTime();

      auto &eval_task_id = community_evaluations_[task_id];
      SendEvaluationTasks(learner_id, eval_task_id);

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
        const auto &metadata = local_tasks_metadata_[learner_id].front();
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
        const auto &metadata = local_tasks_metadata_[learner_id].front();

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

  void SendEvaluateAsync(const std::string &learner_id,
                         const std::string &task_id) {
    auto learner_stub = CreateLearnerStub(learner_id);

    auto &cq = eval_tasks_cq_;
    EvaluateRequest request;
    *request.mutable_task_id() = task_id;
    *request.mutable_model() = model_manager_.GetModel();

    // Call object to store rpc data.
    auto *call = new AsyncLearnerEvalCall;

    // Set the learner id this call will be submitted to.
    call->learner_id = learner_id;

    // stub->PrepareAsyncEvaluate() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    // Opens gRPC channel with learner.
    call->response_reader =
        learner_stub->PrepareAsyncEvaluate(&call->context, request, &cq);

    // Initiate the RPC call.
    call->response_reader->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the memory address of the call
    // object.
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

  void SendTrainAsync(const std::string &learner_id,
                      const std::string &task_id) {
    auto learner_stub = CreateLearnerStub(learner_id);

    auto &cq = run_tasks_cq_;

    TrainRequest request;
    *request.mutable_task_id() = task_id;
    *request.mutable_model() = model_manager_.GetModel();
    *request.mutable_params() = learners_train_params_[learner_id];

    // Call object to store rpc data.
    auto *call = new AsyncLearnerRunTaskCall;

    call->learner_id = learner_id;
    // stub->PrepareAsyncTrain() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    // Opens gRPC channel with learner.
    call->response_reader =
        learner_stub->PrepareAsyncTrain(&call->context, request, &cq);

    // Initiate the RPC call.
    call->response_reader->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the memory address of the call
    // object.
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

  // Implementation of generic AsyncLearnerCall type to handle Train
  // responses.
  struct AsyncLearnerRunTaskCall : AsyncLearnerCall<Ack> {};

  // Implementation of generic AsyncLearnerCall type to handle Evaluate
  // responses.
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
    const ServerParams &server_params,
    const GlobalTrainParams &global_train_params,
    const ModelStoreParams &model_store_params) {
  return absl::make_unique<ControllerDefaultImpl>(
      server_params, global_train_params, model_store_params);
}

}  // namespace metisfl::controller
