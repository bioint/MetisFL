
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
        model_store_params_(move(model_store_params)),
        global_iteration_(0),
        learners_(),
        learners_stub_(),
        learners_train_params_(),
        learners_mutex_(),
        community_model_(),
        scheduling_pool_(2),
        model_store_mutex_(),
        run_tasks_cq_(),
        eval_tasks_cq_() {
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

  const FederatedModel &CommunityModel() const override {
    return community_model_;
  }

  absl::Status SetInitialModel(const Model &model) override {
    PLOG(INFO) << "Received initial model from admin.";
    *community_model_.mutable_model() = model;
    return absl::OkStatus();
  }

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
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    auto it = learners_.find(learner_id);
    if (it != learners_.end()) {
      PLOG(INFO) << "Removing learner from controller: " << learner_id;
      model_store_->EraseModels(std::vector<std::string>{learner_id});
      learners_.erase(it);
      learners_stub_.erase(learner_id);
      learners_train_params_.erase(learner_id);
      return absl::OkStatus();
    } else
      return absl::NotFoundError("Learner is not part of the federation.");
  }

  absl::Status StartTraining() override {
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    // Schedule initial tasks for all learners.
    for (const auto &[learner_id, learner] : learners_) {
      scheduling_pool_.push_task(
          [this, learner_id] { ScheduleInitialTask(learner_id); });
    }

    return absl::OkStatus();
  }

  absl::Status TrainDone(const TrainDoneRequest &task) override {
    auto learner_id = task.learner_id();
    RETURN_IF_ERROR(ValidateLearner(learner_id));

    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    auto task_global_iteration = task.metadata().global_iteration();
    auto metadata_index =
        task_global_iteration == 0 ? 0 : task_global_iteration - 1;

    if (not metadata_.empty() && metadata_index < metadata_.size()) {
      *metadata_.at(metadata_index).add_completed_by_learner_id() = learner_id;
      (*metadata_.at(metadata_index)
            .mutable_train_task_received_at())[learner_id] =
          TimeUtil::GetCurrentTime();
    }

    // Inserts learner's new local model. This is a blocking call.
    model_store_->InsertModel(std::vector<std::pair<std::string, Model>>{
        std::pair<std::string, Model>(learner_id, task.model())});
    PLOG(INFO) << "Inserted learner\'s " << learner_id << " model.";

    // Update learner collection with metrics from last completed training task.
    if (!local_tasks_metadata_.contains(learner_id)) {
      local_tasks_metadata_[learner_id] = std::list<TaskExecutionMetadata>();
    }
    local_tasks_metadata_[learner_id].push_back(task.metadata());

    // Schedules next tasks if necessary. We call ScheduleTasks() asynchronously
    // because during synchronous execution, the learner who completed its local
    // training task the last within a federation round will have to wait for
    // all the rest of training tasks to be scheduled before it can receive an
    // acknowledgement by the controller for its completed task. Put it simply,
    // the learner who completed its task the last within a round will have to
    // keep a connection open with the controller, till the controller schedules
    // all necessary training tasks for the next federation round.
    scheduling_pool_.push_task(
        [this, learner_id, task] { ScheduleTasks(learner_id, task); });

    return absl::OkStatus();
  }

  std::vector<FederatedTaskRuntimeMetadata> GetRuntimeMetadataLineage(
      uint32_t num_steps) override {
    if (metadata_.empty()) {
      return {};
    }
    const auto &lineage = metadata_;
    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<FederatedTaskRuntimeMetadata> lineage_head;
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

  std::vector<TaskExecutionMetadata> GetLocalTaskLineage(
      const std::string &learner_id, uint32_t num_steps) override {
    if (!local_tasks_metadata_.contains(learner_id)) {
      return {};
    }

    const auto &lineage = local_tasks_metadata_[learner_id];

    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<TaskExecutionMetadata> lineage_head;
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
  typedef std::unique_ptr<LearnerService::Stub> LearnerStub;

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

  void ScheduleInitialTask(const std::string &learner_id) {
    if (!community_model_.IsInitialized()) {
      return;
    }

    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    if (metadata_.empty()) {
      FederatedTaskRuntimeMetadata meta = FederatedTaskRuntimeMetadata();
      ++global_iteration_;
      meta.set_global_iteration(global_iteration_);
      *meta.mutable_started_at() = TimeUtil::GetCurrentTime();
      metadata_.emplace_back(meta);

      PLOG(INFO) << "FedIteration: " << unsigned(global_iteration_);
    }

    // When a new learner joins/trains on the initial task, we record
    // all runtime related metadata to the last item in the metadata collection.
    auto &meta = metadata_.back();
    // Records the learner id to which the controller delegates the latest task.
    *meta.add_assigned_to_learner_id() = learner_id;
    auto &community_model = community_model_;

    // Send initial training task.
    std::vector<std::string> learner_to_list_ = {learner_id};
    // We also need to pass the metadata object to record submission time.
    SendRunTasks(learner_to_list_, community_model, meta);
  }

  void ScheduleTasks(const std::string &learner_id,
                     const TrainDoneRequest &task) {
    // TODO(@stripeli): Maybe put a guard for global_iteration_ as well?
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    auto to_schedule = scheduler_->ScheduleNext(learner_id, GetLearnerIds());

    if (!to_schedule.empty()) {
      // Updates completion time of the just completed scheduled task.
      auto task_global_iteration = task.metadata().global_iteration();
      // Assign a non-negative value to the metadata index.
      auto metadata_index =
          (task_global_iteration == 0) ? 0 : task_global_iteration - 1;

      if (not metadata_.empty() && metadata_index < metadata_.size()) {
        *metadata_.at(metadata_index).mutable_completed_at() =
            TimeUtil::GetCurrentTime();
      }

      // Select models that will participate in the community model.
      auto selected_for_aggregation =
          selector_->Select(to_schedule, GetLearnerIds());

      // Computes the community model using models that have
      // been selected by the model selector.
      auto community_model =
          ComputeCommunityModel(selected_for_aggregation, metadata_index);

      // Record the number of zeros and non-zeros values for
      // each model layer/variable in the metadata collection.
      RecordCommunityModelSize(community_model, metadata_index);

      community_model.set_global_iteration(task_global_iteration);
      // Updates the community model.
      community_model_ = community_model;

      // Creates an evaluation hash map container for the new community model.
      CommunityModelEvaluation community_eval;
      // Records the evaluation of the community model that was
      // computed at the previously completed global iteration.
      community_eval.set_global_iteration(task_global_iteration);
      community_evaluations_.push_back(community_eval);

      // Evaluate the community model across all scheduled learners. Send
      // community model evaluation tasks to all scheduled learners. Each
      // learner holds a different training, validation and test dataset
      // and hence we need to evaluate the community model over each dataset.
      // All evaluation tasks are submitted asynchronously, therefore, to
      // make sure that all metrics are properly collected when an EvaluateModel
      // response is received, we pass the index of the `community_evaluations_`
      // vector of the position to which the current community model refers to.
      // We also pass the index to the `metadata_` vector to which the current
      // evaluation task corresponds and needs to store the associated meta
      // data.
      SendEvaluationTasks(to_schedule, community_model,
                          community_evaluations_.size() - 1, metadata_index);

      // Increase global iteration counter to reflect the new scheduling round.
      ++global_iteration_;
      PLOG(INFO) << "FedIteration: " << unsigned(global_iteration_);

      // Set the specifications of the next training task.
      UpdateLearnersTaskTemplates(to_schedule);

      // Creates a new federation runtime metadata
      // object for the new scheduling round.
      FederatedTaskRuntimeMetadata new_meta = FederatedTaskRuntimeMetadata();
      new_meta.set_global_iteration(global_iteration_);
      *new_meta.mutable_started_at() = TimeUtil::GetCurrentTime();
      // Records the id of the learners to which
      // the controller delegates the training task.
      for (const auto &to_schedule_id : to_schedule) {
        *new_meta.add_assigned_to_learner_id() = to_schedule_id;
      }

      PLOG(INFO) << "Sending training tasks";
      // Send training task to all scheduled learners.
      SendRunTasks(to_schedule, community_model, new_meta);
      PLOG(INFO) << "Sent training tasks";

      // Save federated task runtime metadata.
      metadata_.emplace_back(new_meta);
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

  void SendEvaluationTasks(std::vector<std::string> &learners,
                           const FederatedModel &model,
                           const uint32_t &comm_eval_ref_idx,
                           const uint32_t &metadata_ref_idx) {
    // The implementation follows the (recommended) async
    // grpc client:
    // https://github.com/grpc/grpc/blob/master/examples/cpp/helloworld/greeter_async_client2.cc

    for (const auto &learner_id : learners) {
      (*metadata_.at(metadata_ref_idx)
            .mutable_eval_task_submitted_at())[learner_id] =
          TimeUtil::GetCurrentTime();
      SendEvaluateAsync(learner_id, model, comm_eval_ref_idx, metadata_ref_idx);
    }
  }

  void SendEvaluateAsync(const std::string &learner_id,
                         const FederatedModel &model,
                         const uint32_t &comm_eval_ref_idx,
                         const uint32_t &metadata_ref_idx) {
    // FIXME(stripeli,canastas): This needs to be reimplemented by using
    //  a single channel or stub. We tried to implement this that way, but when
    //  we run either of the two approaches either through the (reused) channel
    //  or stub, the requests were delayed substantially and not received
    //  immediately by the learners. For instance, under normal conditions
    //  sending the tasks should take around 10-20secs but by reusing the
    //  grpc Stub/Channel sending all requests would take around 100-120secs.
    //  This behavior did not occur when testing with small model proto
    //  messages, e.g., DenseNet FashionMNIST with 120k params but it is clearly
    //  evident when working with very large model proto messages, e.g.,
    //  CIFAR-10 with 1.6M params and encryption using FHE ~ 100MBs per model.
    auto learner_stub = CreateLearnerStub(learner_id);

    auto &cq = eval_tasks_cq_;
    EvaluateRequest request;
    *request.mutable_model() = model.model();

    // Call object to store rpc data.
    auto *call = new AsyncLearnerEvalCall;

    // Set the learner id this call will be submitted to.
    call->learner_id = learner_id;

    // Set the index of the community model evaluation vector evaluation
    // metrics and values will be stored when response is received.
    call->comm_eval_ref_idx = comm_eval_ref_idx;

    // Set the index of the global model iteration to access the metadata_
    // vector collection and store additional metadata related to the
    // evaluation request, such as reception time of the evaluation task.
    call->metadata_ref_idx = metadata_ref_idx;

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

  void SendRunTasks(std::vector<std::string> &learners,
                    const FederatedModel &model,
                    FederatedTaskRuntimeMetadata &meta) {
    for (const auto &learner_id : learners) {
      (*meta.mutable_train_task_submitted_at())[learner_id] =
          TimeUtil::GetCurrentTime();
      SendTrainAsync(learner_id, model);
    }
  }

  void SendTrainAsync(const std::string &learner_id,
                      const FederatedModel &model) {
    auto learner_stub = CreateLearnerStub(learner_id);

    auto &cq = run_tasks_cq_;
    auto global_iteration =
        global_iteration_;  // FIXME: need to figure out global iteration

    TrainRequest request;
    *request.mutable_model() = model.model();
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

  FederatedModel ComputeCommunityModel(
      const std::vector<std::string> &learners_ids,
      const uint32_t &metadata_ref_idx) {
    // Handles the case where the community model is requested for the
    // first time and has the original (random) initialization state.
    if (global_iteration_ < 1 && community_model_.IsInitialized()) {
      return community_model_;
    }

    // We need to lock the model store to avoid reading stale models.
    // Basically, we need to make sure that all local models used
    // for aggregation are valid pointers, if we do not lock then
    // unexpected behaviors occur and the controller crashes.
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    *metadata_.at(metadata_ref_idx).mutable_model_aggregation_started_at() =
        TimeUtil::GetCurrentTime();
    auto start_time_aggregation = std::chrono::high_resolution_clock::now();

    FederatedModel new_community_model;  // return variable.

    // Select a sub-set of learners who are participating in the experiment.
    // The selection needs to be a reference to LearnerDescriptor to avoid copy.
    absl::flat_hash_map<std::string, LearnerDescriptor *> participating_states;
    absl::flat_hash_map<std::string, TaskExecutionMetadata *>
        participating_metadata;

    // While looping over the learners collection, we need to make sure:
    // (1)  All learners exist in the learners_ collection. We do so to avoid
    // cases
    //      where a learner was removed (dropped) from the collection.
    // (2)  All learners' models used for the aggregation have been committed.
    //      In other words, there is at least one (local) model for every
    //      learner.
    for (const auto &id : learners_ids) {
      if (learners_.contains(id) && local_tasks_metadata_.contains(id)) {
        participating_states[id] = &learners_.at(id);
        participating_metadata[id] = &local_tasks_metadata_.at(id).back();
      }
    }

    // Before performing any aggregation, we need first to compute the
    // normalized scaling factor or contribution value of each model in
    // the community/global/aggregated model.
    auto scaling_factors = scaler_->ComputeScalingFactors(
        community_model_, learners_, participating_states,
        participating_metadata);

    // Defines the length of the aggregation stride, i.e., how many models
    // to fetch from the model store and feed to the aggregation function.
    // Only FedStride does this stride-based aggregation. All other aggregation
    // rules use the entire list of participating models.
    uint32_t aggregation_stride_length = participating_states.size();
    if (global_train_params_.aggregation_rule == "FedStride") {
      auto fed_stride_length = global_train_params_.stride_length;
      if (fed_stride_length > 0) {
        aggregation_stride_length = fed_stride_length;
      }
    }

    /* Since absl does not support crbeing() or iterator decrement (--) we need
       to use this. method to find the itr of the last element. */
    absl::flat_hash_map<std::string, LearnerDescriptor *>::iterator
        last_elem_itr;
    for (auto itr = participating_states.begin();
         itr != participating_states.end(); itr++) {
      last_elem_itr = itr;
    }

    std::vector<std::pair<std::string, int>>
        to_select_block;  // e.g., { (learner_id, stride_length), ...}
    std::vector<std::vector<std::pair<const Model *, double>>>
        to_aggregate_block;  // e.g., { {m1*, 0.1}, {m2*, 0.3}, ...}
    std::vector<std::pair<const Model *, double>>
        to_aggregate_learner_models_tmp;
    for (auto itr = participating_states.begin();
         itr != participating_states.end(); itr++) {
      auto const &learner_id = itr->first;

      // This represents the number of models to be fetched from the back-end.
      // We need to check if the back-end has stored more models than the
      // required model number of the aggregation strategy.
      const auto learner_lineage_length =
          model_store_->GetLearnerLineageLength(learner_id);
      int select_lineage_length =
          (learner_lineage_length >=
           aggregator_->RequiredLearnerLineageLength())
              ? aggregator_->RequiredLearnerLineageLength()
              : learner_lineage_length;
      to_select_block.emplace_back(learner_id, select_lineage_length);

      uint32_t block_size = to_select_block.size();
      if (block_size == aggregation_stride_length || itr == last_elem_itr) {
        PLOG(INFO) << "Computing for block size: " << block_size;
        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_size()
             ->Add() = block_size;

        /*! --- SELECT MODELS ---
         * Here, we retrieve models from the back-end model store.
         * We need to import k-number of models from the model store.
         * Number k depends on the number of models required by the aggregator
         * or the number of local models stored for each learner, whichever is
         * smaller.
         *
         *  Case (1): Redis Store: we select models from an outside (external)
         * store. Case (2): In-Memory Store: we select models from the in-memory
         * hash map.
         *
         *  In both cases, a pointer would be returned for the models stored in
         * the model store.
         */
        auto start_time_selection = std::chrono::high_resolution_clock::now();
        std::map<std::string, std::vector<const Model *>> selected_models =
            model_store_->SelectModels(to_select_block);
        auto end_time_selection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time_selection =
            end_time_selection - start_time_selection;
        auto avg_time_selection_per_model =
            elapsed_time_selection.count() / block_size;
        for (auto const &[selected_learner_id, selected_learner_models] :
             selected_models) {
          (*metadata_.at(metadata_ref_idx)
                .mutable_model_selection_duration_ms())[selected_learner_id] =
              avg_time_selection_per_model;
        }

        /* --- CONSTRUCT MODELS TO AGGREGATE --- */
        for (auto const &[selected_learner_id, selected_learner_models] :
             selected_models) {
          auto scaling_factor = scaling_factors[selected_learner_id];
          for (auto it : selected_learner_models) {
            to_aggregate_learner_models_tmp.emplace_back(it, scaling_factor);
          }
          to_aggregate_block.push_back(to_aggregate_learner_models_tmp);
          to_aggregate_learner_models_tmp.clear();
        }

        /* --- AGGREGATE MODELS --- */
        auto start_time_block_aggregation =
            std::chrono::high_resolution_clock::now();
        // FIXME(@stripeli): When using Redis as backend and setting to
        //  store all models (i.e., EvictionPolicy = NoEviction) then
        //  the collection of models passed here are all models and
        //  therefore the number of variables becomes inconsistent with the
        //  number of variables of the original model. For instance, if a
        //  learner has two models stored in the collection and each model
        //  has 6 variables then the total variables of the sampled model
        //  will be 12 not 6 (the expected)!. Try the following for testing:
        //    auto total_sampled_vars = sample_model->variables_size();
        //    PLOG(INFO) << "TOTAL SAMPLED VARS:" << total_sampled_vars;
        new_community_model = aggregator_->Aggregate(to_aggregate_block);
        auto end_time_block_aggregation =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli>
            elapsed_time_block_aggregation =
                end_time_block_aggregation - start_time_block_aggregation;
        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_duration_ms()
             ->Add() = elapsed_time_block_aggregation.count();

        long block_memory = GetTotalMemory();
        PLOG(INFO) << "Aggregate block memory usage (kb): " << block_memory;
        *metadata_.at(metadata_ref_idx)
             .mutable_model_aggregation_block_memory_kb()
             ->Add() = (double)block_memory;

        // Cleanup. Clear sentinel block variables and reset
        // model_store's state to reclaim unused memory.
        to_select_block.clear();
        to_aggregate_block.clear();
        model_store_->ResetState();

      }  // end-if

    }  // end for loop

    // Reset aggregation function's state for the next step.
    aggregator_->Reset();

    // Compute elapsed time for the entire aggregation - global model
    // computation function.
    auto end_time_aggregation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_aggregation =
        end_time_aggregation - start_time_aggregation;
    metadata_.at(metadata_ref_idx)
        .set_model_aggregation_total_duration_ms(
            elapsed_time_aggregation.count());
    *metadata_.at(metadata_ref_idx).mutable_model_aggregation_completed_at() =
        TimeUtil::GetCurrentTime();

    return new_community_model;
  }

  void RecordCommunityModelSize(const FederatedModel &model,
                                const uint32_t &metadata_ref_idx) {
    for (auto &tensor : model.model().tensors()) {
      if (!tensor.encrypted()) {
        metisfl::proto::ProtoSerde<> proto_serde =
            metisfl::proto::GetProtoSerde(tensor.type().type());

        TensorQuantifier tensor_quantifier = proto_serde.QuantifyTensor(tensor);

        *metadata_.at(metadata_ref_idx)
             .mutable_model_tensor_quantifiers()
             ->Add() = tensor_quantifier;
      } else {
        auto tensor_quantifier = metisfl::TensorQuantifier();
        tensor_quantifier.set_tensor_size_bytes(tensor.ByteSizeLong());
        *metadata_.at(metadata_ref_idx)
             .mutable_model_tensor_quantifiers()
             ->Add() = tensor_quantifier;
      }
    }
  }

  uint32_t global_iteration_;
  ServerParams server_params_;
  GlobalTrainParams global_train_params_;
  ModelStoreParams model_store_params_;

  FederatedModel community_model_;
  std::vector<FederatedTaskRuntimeMetadata> metadata_;
  std::vector<CommunityModelEvaluation> community_evaluations_;

  absl::flat_hash_map<std::string, LearnerDescriptor> learners_;
  absl::flat_hash_map<std::string, LearnerStub> learners_stub_;
  absl::flat_hash_map<std::string, TrainParams> learners_train_params_;
  absl::flat_hash_map<std::string, std::list<TaskExecutionMetadata>>
      local_tasks_metadata_;

  std::mutex learners_mutex_;
  std::mutex model_store_mutex_;
  BS::thread_pool scheduling_pool_;

  std::unique_ptr<AggregationFunction> aggregator_ =
      CreateAggregator(global_train_params_);
  std::unique_ptr<ModelStore> model_store_ =
      CreateModelStore(model_store_params_);
  std::unique_ptr<ScalingFunction> scaler_ =
      CreateScaler(global_train_params_.scaling_factor);
  std::unique_ptr<Scheduler> scheduler_ =
      CreateScheduler(global_train_params_.communication_protocol);
  std::unique_ptr<Selector> selector_ = CreateSelector();

  grpc::CompletionQueue run_tasks_cq_;
  grpc::CompletionQueue eval_tasks_cq_;

  template <typename T>
  struct AsyncLearnerCall {
    std::string learner_id;
    T reply;
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<T>> response_reader;
  };

  // Implementation of generic AsyncLearnerCall type to handle RunTask
  // responses.
  struct AsyncLearnerRunTaskCall : AsyncLearnerCall<Ack> {};

  // Implementation of generic AsyncLearnerCall type to handle EvaluateModel
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
