
#include <mutex>
#include <utility>
#include <thread>

#include <glog/logging.h>
#include <google/protobuf/util/time_util.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/create_channel.h>

#include "absl/memory/memory.h"
#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/learner.grpc.pb.h"
#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {
namespace {

using google::protobuf::util::TimeUtil;

class ControllerDefaultImpl : public Controller {
 public:
  ControllerDefaultImpl(ControllerParams &&params,
                        std::unique_ptr<ScalingFunction> scaler,
                        std::unique_ptr<AggregationFunction> aggregator,
                        std::unique_ptr<Scheduler> scheduler,
                        std::unique_ptr<Selector> selector,
                        std::unique_ptr<ModelStore> model_store)
      : params_(std::move(params)), global_iteration_(0), learners_(),
        learners_stub_(), learners_task_template_(), learners_mutex_(),
        scaler_(std::move(scaler)), aggregator_(std::move(aggregator)),
        scheduler_(std::move(scheduler)), selector_(std::move(selector)),
        community_model_(), scheduling_pool_(2),
        model_store_(std::move(model_store)), model_store_mutex_(),
        run_tasks_cq_(), eval_tasks_cq_() {

    // We perform the following detachment because we want to have only
    // one thread and one completion queue to handle asynchronous request
    // submission and digestion. In the previous implementation, we were
    // always spawning a new thread for every run task request and a new
    // thread for every evaluate model request. With the refactoring,
    // there is only one thread to handle all SendRunTask requests
    // submission, one thread to handle all EvaluateModel requests
    // submission and one thread to digest SendRunTask responses
    // and one thread to digest EvaluateModel responses.

    // One thread to handle learners' responses to RunTasks requests.
    std::thread run_tasks_digest_t_(
        &ControllerDefaultImpl::DigestRunTasksResponses, this);
    run_tasks_digest_t_.detach();

    // One thread to handle learners' responses to EvaluateModel requests.
    std::thread eval_tasks_digest_t_(
        &ControllerDefaultImpl::DigestEvaluationTasksResponses, this);
    eval_tasks_digest_t_.detach();

  }

  const ControllerParams &GetParams() const override { return params_; }

  std::vector<LearnerDescriptor> GetLearners() const override {

    // TODO(stripeli): Shall we 'hide' authentication token from exposure?
    std::vector<LearnerDescriptor> learners;
    for (const auto &[key, learner_state]: learners_) {
      learners.push_back(learner_state.learner());
    }
    return learners;

  }

  uint32_t GetNumLearners() const override { return learners_.size(); }

  const FederatedModel &CommunityModel() const override {
    return community_model_;
  }

  // TODO(stripeli): add admin auth token support for replacing model.
  absl::Status
  ReplaceCommunityModel(const FederatedModel &model) override {

    // When replacing the federation model we only set the number of learners
    // contributed to this model and the actual model. We do not replace the
    // global iteration since it is updated exclusively by the controller.
    PLOG(INFO) << "Replacing community model.";
    community_model_.set_num_contributors(model.num_contributors());
    *community_model_.mutable_model() = model.model();
    return absl::OkStatus();

  }

  absl::StatusOr<LearnerDescriptor>
  AddLearner(const ServerEntity &server_entity,
             const DatasetSpec &dataset_spec) override {

    // Acquires a lock to avoid having multiple threads overwriting the learners'
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    // Validates non-empty hostname and non-negative port.
    if (server_entity.hostname().empty() || server_entity.port() < 0) {
      return absl::InvalidArgumentError("Hostname and port must be provided.");
    }

    // Validates number of train, validation and test examples. Train examples
    // must always be positive, while validation and test can be non-negative.
    if (dataset_spec.num_training_examples() <= 0) {
      return absl::InvalidArgumentError("Learner training examples <= 0.");
    }

    // TODO(stripeli): Condition to ping the connected learner (hostname:port).

    // Generates learner id.
    const std::string learner_id = GenerateLearnerId(server_entity);

    if (learners_.contains(learner_id)) {
      // Learner was already registered with the controller.
      return absl::AlreadyExistsError("Learner has already joined.");
    }

    // Generates an auth token for the learner.
    // TODO(stripeli) We need a better authorization token generator.
    const std::string auth_token = std::to_string(learners_.size() + 1);

    // Initializes learner state with an empty model.
    LearnerDescriptor learner;
    learner.set_id(learner_id);
    learner.set_auth_token(auth_token);
    *learner.mutable_server_entity() = server_entity;
    *learner.mutable_dataset_spec() = dataset_spec;

    LearnerState learner_state;
    *learner_state.mutable_learner() = learner;

    // Creates default task template.
    LearningTaskTemplate task_template;
    // Make sure steps per epoch is always positive. For instance if
    // the dataset size is less than the batch size, then the steps will
    // be equal to 0; hence the ceiling operation and float conversion.
    // Float conversion because ceil(x/y) with x < y and x, y integers returns 0.
    uint32_t steps_per_epoch = std::ceil(
        (float) dataset_spec.num_training_examples() /
            (float) params_.model_hyperparams().batch_size());

    task_template.set_num_local_updates(
      params_.model_hyperparams().epochs() * steps_per_epoch);

    // Registers learner.
    learners_[learner_id] = learner_state;
    learners_task_template_[learner_id] = task_template;

    // Opens gRPC connection with the learner.
    learners_stub_[learner_id] = CreateLearnerStub(learner_id);

    // Triggers the initial task.
    scheduling_pool_.push_task(
        [this, learner_id] { ScheduleInitialTask(learner_id); });

    return learner;

  }

  absl::Status
  RemoveLearner(const std::string &learner_id,
                const std::string &token) override {

    RETURN_IF_ERROR(ValidateLearner(learner_id, token));

    // Acquires a lock to avoid having multiple threads overwriting the learners
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    auto it = learners_.find(learner_id);
    // Checks requesting learner existence inside the state map.
    if (it != learners_.end()) {
      if (it->second.learner().auth_token() == token) {
        PLOG(INFO) << "Removing learner from controller: " << learner_id;
        model_store_->EraseModels(std::vector<std::string>{learner_id});
        learners_.erase(it);
        learners_stub_.erase(learner_id);
        learners_task_template_.erase(learner_id);
        return absl::OkStatus();
      } else {
        return absl::UnauthenticatedError("Learner token is wrong.");
      }
    } else {
      return absl::NotFoundError("Learner is not part of the federation.");
    }

  }

  absl::Status
  LearnerCompletedTask(const std::string &learner_id, const std::string &token,
                       const CompletedLearningTask &task) override {

    RETURN_IF_ERROR(ValidateLearner(learner_id, token));

    // We need to lock the model store when receiving a new local model.
    // The reason is that there are cases where, a learner might update
    // his previous model, but that previous model is already being used
    // for aggregation. Such a race will lead to null pointer exception
    // during local models aggregation.
    std::lock_guard<std::mutex> model_store_guard(model_store_mutex_);

    // Assign a non-negative value to the metadata index.
    auto task_global_iteration = task.execution_metadata().global_iteration();
    auto metadata_index =
        task_global_iteration == 0 ? 0 : task_global_iteration - 1;
    // Records the id of the learner completed the task.
    if (not metadata_.empty() && metadata_index < metadata_.size()) {
      *metadata_.at(metadata_index).add_completed_by_learner_id() = learner_id;
      (*metadata_.at(metadata_index).mutable_train_task_received_at())[learner_id] =
          TimeUtil::GetCurrentTime();
    }

    // Inserts learner's new local model. This is a blocking
    // call and the reason is that we need learner's local
    // model stored inside the model store before any aggregation
    // operation can happen.
    // Future thoughts on multi-threading insertions.
    //  (1) In the case of InMemory store, we can perform multi-threading,
    //      since we are using a vector to insert learners models.
    //  (2) In the case of Redis, we cannot perform multi-threading,
    //      since Redis is single-thread.
    PLOG(INFO) << "Inserting learner\'s " << learner_id << " model.";
    model_store_->InsertModel(
        std::vector<std::pair<std::string, Model>> {
            std::pair<std::string, Model>(learner_id, task.model())
        });
    PLOG(INFO) << "Inserted learner\'s " << learner_id << " model.";

    // Update learner collection with metrics from last completed training task.
    if (!local_tasks_metadata_.contains(learner_id)) {
      local_tasks_metadata_[learner_id] = std::list<TaskExecutionMetadata>();
    }
    local_tasks_metadata_[learner_id].push_back(task.execution_metadata());

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

  std::vector<FederatedTaskRuntimeMetadata>
  GetRuntimeMetadataLineage(uint32_t num_steps) override {

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

  std::vector<CommunityModelEvaluation>
  GetEvaluationLineage(uint32_t num_steps) override {

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

  std::vector<TaskExecutionMetadata>
  GetLocalTaskLineage(const std::string &learner_id,
                      uint32_t num_steps) override {

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

    // Proper shutdown of the controller process.
    // Send shutdown signal to the completion queues and
    // gracefully close the scheduling pool.
    run_tasks_cq_.Shutdown();
    eval_tasks_cq_.Shutdown();
    scheduling_pool_.wait_for_tasks();
    model_store_->Shutdown();

  }

 private:
  typedef std::unique_ptr<LearnerService::Stub> LearnerStub;

  LearnerStub CreateLearnerStub(const std::string &learner_id) {

    auto server_entity = learners_[learner_id].learner().server_entity();
    auto target =
        absl::StrCat(server_entity.hostname(), ":", server_entity.port());

    auto creds = grpc::InsecureChannelCredentials();
    if (server_entity.has_ssl_config()) {
        grpc::SslCredentialsOptions ssl_opts;
        if (server_entity.ssl_config().enable()) {
          if (server_entity.ssl_config().has_ssl_config_stream()) {
            ssl_opts.pem_root_certs =
                server_entity.ssl_config().ssl_config_stream().public_certificate_stream();
            creds = grpc::SslCredentials(ssl_opts);
          } else {
            PLOG(WARNING) << "Even though learner: " << learner_id <<
            "has requested TLS/SSL connection, it has not sent a public "
            "certificate (as a stream) to establish connection.";
          }
        }
    }
    auto channel = grpc::CreateChannel(target, creds);
    return LearnerService::NewStub(channel);
  }

  absl::Status ValidateLearner(const std::string &learner_id,
                               const std::string &token) const {

    // Validates non-empty learner_id and authentication token.
    if (learner_id.empty() || token.empty()) {
      return absl::InvalidArgumentError("Learner id and token cannot be empty");
    }

    const auto &learner = learners_.find(learner_id);
    if (learner == learners_.end()) {
      return absl::NotFoundError("Learner does not exist.");
    } else if (learner->second.learner().auth_token() != token) {
      return absl::PermissionDeniedError("Invalid token provided.");
    }

    return absl::OkStatus();

  }

  void ScheduleInitialTask(const std::string &learner_id) {

    if (!community_model_.IsInitialized()) {
      return;
    }

    std::lock_guard<std::mutex> learners_guard(learners_mutex_);

    if (metadata_.empty()) {
      // When the very first local training task is scheduled, we need to
      // increase the global iteration counter and create the first
      // federation runtime metadata object.
      FederatedTaskRuntimeMetadata meta = FederatedTaskRuntimeMetadata();
      ++global_iteration_;
      PLOG(INFO) << "FedIteration: " << unsigned(global_iteration_);
      meta.set_global_iteration(global_iteration_);
      *meta.mutable_started_at() = TimeUtil::GetCurrentTime();
      metadata_.emplace_back(meta);
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
                     const CompletedLearningTask &task) {

    // Acquires a lock to avoid having multiple threads overwriting the learners
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    // TODO(@stripeli): Maybe put a guard for global_iteration_ as well?
    std::lock_guard<std::mutex> learners_guard(learners_mutex_);
  
    auto to_schedule =
        scheduler_->ScheduleNext(learner_id, task, GetLearners());
    if (!to_schedule.empty()) {
      // Updates completion time of the just completed scheduled task.
      auto task_global_iteration = task.execution_metadata().global_iteration();
      // Assign a non-negative value to the metadata index.
      auto metadata_index =
          (task_global_iteration == 0) ? 0 : task_global_iteration - 1;

      if (not metadata_.empty() && metadata_index < metadata_.size()) {
        *metadata_.at(metadata_index).mutable_completed_at() =
            TimeUtil::GetCurrentTime();
      }

      // Select models that will participate in the community model.
      auto selected_for_aggregation =
          selector_->Select(to_schedule, GetLearners());

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
      // evaluation task corresponds and needs to store the associated meta data.
      SendEvaluationTasks(to_schedule,
                          community_model,
                          community_evaluations_.size() - 1,
                          metadata_index);

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
      for (const auto &to_schedule_id: to_schedule) {
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
    const auto &communication_specs = params_.communication_specs();
    const auto &protocol_specs = communication_specs.protocol_specs();
    // We check if it is the 2nd global_iteration_, because the 1st
    // global_iteration_ refers to the very first initially scheduled task.
    // For the SemiSynchronous protocol, recomputation takes place only
    // if we are just starting the 2nd federation round or we want to 
    // recompute the number of updates at every federation round 
    // (i.e., dynamic assignment of local training steps).
    if (communication_specs.protocol() ==
        CommunicationSpecs::SEMISYNCHRONOUS &&
        (global_iteration_ == 2 ||
            protocol_specs.semi_sync_recompute_num_updates())) {

      // Finds the slowest learner.
      // float ms_per_batch_slowest = std::numeric_limits<float>::min();
      float ms_per_epoch_slowest = std::numeric_limits<float>::min();
      for (const auto &learner_id: learners) {
        const auto &metadata = local_tasks_metadata_[learner_id].front();
        // if (metadata.processing_ms_per_batch() > ms_per_batch_slowest) {
        //   ms_per_batch_slowest = metadata.processing_ms_per_batch();
        // }
        if (metadata.processing_ms_per_epoch() > ms_per_epoch_slowest) {
          ms_per_epoch_slowest = metadata.processing_ms_per_epoch();
        }
      }

      // Calculates the allowed time for training.
      float t_max = static_cast<float>(
          protocol_specs.semi_sync_lambda()) *
          ms_per_epoch_slowest;

      // Updates the task templates based on the slowest learner.
      for (const auto &learner_id: learners) {
        const auto &metadata = local_tasks_metadata_[learner_id].front();

        auto processing_ms_per_batch = metadata.processing_ms_per_batch();
        if (processing_ms_per_batch == 0) {
          PLOG(ERROR) << "Processing ms per batch is zero. Setting to 1.";
          processing_ms_per_batch = 1;
        }
        int num_local_updates = std::ceil(t_max / processing_ms_per_batch);

        auto &task_template = learners_task_template_[learner_id];
        task_template.set_num_local_updates(num_local_updates);
      }

    } // end-if.

  }

  void SendEvaluationTasks(std::vector<std::string> &learners,
                           const FederatedModel &model,
                           const uint32_t &comm_eval_ref_idx,
                           const uint32_t &metadata_ref_idx) {

    // Our goal is to send the EvaluateModel request to each learner in parallel.
    // We use a single thread to asynchronously send all evaluate model requests
    // and add every submitted EvaluateModel request inside a grpc::CompletionQueue.
    // The implementation follows the (recommended) async grpc client:
    // https://github.com/grpc/grpc/blob/master/examples/cpp/helloworld/greeter_async_client2.cc
    for (const auto &learner_id: learners) {
      (*metadata_.at(metadata_ref_idx)
          .mutable_eval_task_submitted_at())[learner_id] = TimeUtil::GetCurrentTime();
      SendEvaluationTaskAsync(
          learner_id, model, comm_eval_ref_idx, metadata_ref_idx);
    }
  }

  void SendEvaluationTaskAsync(const std::string &learner_id,
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
    auto &params = params_;
    EvaluateModelRequest request;
    *request.mutable_model() = model.model();
    request.set_batch_size(params.model_hyperparams().batch_size());
    request.add_evaluation_dataset(EvaluateModelRequest::TRAINING);
    request.add_evaluation_dataset(EvaluateModelRequest::VALIDATION);
    request.add_evaluation_dataset(EvaluateModelRequest::TEST);

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

    // stub->PrepareAsyncEvaluateModel() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    // Opens gRPC channel with learner.
    call->response_reader = learner_stub->
        PrepareAsyncEvaluateModel(&call->context, request, &cq);

    // Initiate the RPC call.
    call->response_reader->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the memory address of the call object.
    call->response_reader->Finish(&call->reply, &call->status, (void *) call);

  }

  void DigestEvaluationTasksResponses() {

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
              .mutable_eval_task_received_at())[call->learner_id] = TimeUtil::GetCurrentTime();
          ModelEvaluations model_evaluations;
          *model_evaluations.mutable_training_evaluation() =
              call->reply.evaluations().training_evaluation();
          *model_evaluations.mutable_validation_evaluation() =
              call->reply.evaluations().validation_evaluation();
          *model_evaluations.mutable_test_evaluation() =
              call->reply.evaluations().test_evaluation();
          (*community_evaluations_.at(call->comm_eval_ref_idx)
              .mutable_evaluations())[call->learner_id] = model_evaluations;
        } else {
          PLOG(ERROR) << "EvaluateModel RPC request to learner: " << call->learner_id
                      << " failed with error: " << call->status.error_message();
        }
      } //end if call

      delete call;

    } // end of loop

  }

  void SendRunTasks(std::vector<std::string> &learners,
                    const FederatedModel &model,
                    FederatedTaskRuntimeMetadata &meta) {

    // Our goal is to send the RunTask request to each learner in parallel.
    // We use a single thread to asynchronously send all run task requests
    // and add every submitted RunTask request inside a grpc::CompletionQueue.
    // The implementation follows the (recommended) async grpc client:
    // https://github.com/grpc/grpc/blob/master/examples/cpp/helloworld/greeter_async_client2.cc
    for (const auto &learner_id: learners) {
      (*meta.mutable_train_task_submitted_at())[learner_id] =
          TimeUtil::GetCurrentTime();
      SendRunTaskAsync(learner_id, model);
    }

  }

  void SendRunTaskAsync(const std::string &learner_id,
                        const FederatedModel &model) {

    auto learner_stub = CreateLearnerStub(learner_id);

    auto &cq = run_tasks_cq_;
    auto &params = params_;
    auto global_iteration = global_iteration_;
    const auto &task_template = learners_task_template_[learner_id];

    RunTaskRequest request;
    *request.mutable_federated_model() = model;
    auto *next_task = request.mutable_task();
    next_task->set_global_iteration(global_iteration);
    const auto &model_params = params.model_hyperparams();
    next_task->set_num_local_updates(
        task_template.num_local_updates()); // get from task template.
    next_task->set_training_dataset_percentage_for_stratified_validation(
        model_params.percent_validation());
    // TODO(stripeli): Add evaluation metrics for the learning task.

    auto *hyperparams = request.mutable_hyperparameters();
    hyperparams->set_batch_size(model_params.batch_size());
    *hyperparams->mutable_optimizer() =
        params.model_hyperparams().optimizer();

    // Call object to store rpc data.
    auto *call = new AsyncLearnerRunTaskCall;

    call->learner_id = learner_id;
    // stub->PrepareAsyncRunTask() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    // Opens gRPC channel with learner.
    call->response_reader = learner_stub->
        PrepareAsyncRunTask(&call->context, request, &cq);

    // Initiate the RPC call.
    call->response_reader->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the memory address of the call object.
    call->response_reader->Finish(&call->reply, &call->status, (void *) call);

  }

  void DigestRunTasksResponses() {

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
      } //end if call

      delete call;

    } // end of loop

  }

  FederatedModel
  ComputeCommunityModel(
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

    FederatedModel new_community_model; // return variable.

    // Select a sub-set of learners who are participating in the experiment.
    // The selection needs to be a reference to learnerState to avoid copy.
    // The LearnerState does not contain any models.
    // All required models are retrieved from the model store.
    absl::flat_hash_map<std::string, LearnerState *> participating_states;
    absl::flat_hash_map<std::string, TaskExecutionMetadata *> participating_metadata;

    // While looping over the learners collection, we need to make sure:
    // (1)  All learners exist in the learners_ collection. We do so to avoid cases
    //      where a learner was removed (dropped) from the collection.
    // (2)  All learners' models used for the aggregation have been committed. 
    //      In other words, there is at least one (local) model for every learner.
    for (const auto &id: learners_ids) {
      if (learners_.contains(id) && local_tasks_metadata_.contains(id)) {
        participating_states[id] = &learners_.at(id);
        participating_metadata[id] = &local_tasks_metadata_.at(id).back();
      }
    }

    // Before performing any aggregation, we need first to compute the
    // normalized scaling factor or contribution value of each model in
    // the community/global/aggregated model.
    auto scaling_factors =
        scaler_->ComputeScalingFactors(
            community_model_, learners_, participating_states, participating_metadata);
            
    // Defines the length of the aggregation stride, i.e., how many models
    // to fetch from the model store and feed to the aggregation function.
    // Only FedStride does this stride-based aggregation. All other aggregation
    // rules use the entire list of participating models.
    uint32_t aggregation_stride_length = participating_states.size();
    if (params_.global_model_specs().aggregation_rule().has_fed_stride()) {
      auto fed_stride_length =
          params_.global_model_specs().aggregation_rule().fed_stride().stride_length();
      if (fed_stride_length > 0) {
        aggregation_stride_length = fed_stride_length;
      }
    }

    /* Since absl does not support crbeing() or iterator decrement (--) we need to use this.
       method to find the itr of the last element. */
    absl::flat_hash_map<std::string, LearnerState *>::iterator last_elem_itr;
    for (auto itr = participating_states.begin(); itr != participating_states.end(); itr++) {
      last_elem_itr = itr;
    }

    std::vector<std::pair<std::string, int>> to_select_block; // e.g., { (learner_id, stride_length), ...}
    std::vector<std::vector<std::pair<const Model *, double>>>
        to_aggregate_block; // e.g., { {m1*, 0.1}, {m2*, 0.3}, ...}
    std::vector<std::pair<const Model *, double>> to_aggregate_learner_models_tmp;
    for (auto itr = participating_states.begin(); itr != participating_states.end(); itr++) {

      auto const &learner_id = itr->first;

      // This represents the number of models to be fetched from the back-end.
      // We need to check if the back-end has stored more models than the
      // required model number of the aggregation strategy.
      const auto learner_lineage_length =
          model_store_->GetLearnerLineageLength(learner_id);
      int select_lineage_length =
          (learner_lineage_length >= aggregator_->RequiredLearnerLineageLength())
          ? aggregator_->RequiredLearnerLineageLength() : learner_lineage_length;
      to_select_block.emplace_back(learner_id, select_lineage_length);

      uint32_t block_size = to_select_block.size();
      if (block_size == aggregation_stride_length || itr == last_elem_itr) {

        PLOG(INFO) << "Computing for block size: " << block_size;
        *metadata_.at(metadata_ref_idx).mutable_model_aggregation_block_size()->Add() = block_size;

        /*! --- SELECT MODELS ---
         * Here, we retrieve models from the back-end model store.
         * We need to import k-number of models from the model store.
         * Number k depends on the number of models required by the aggregator or
         * the number of local models stored for each learner, whichever is smaller.
         *
         *  Case (1): Redis Store: we select models from an outside (external) store.
         *  Case (2): In-Memory Store: we select models from the in-memory hash map.
         *
         *  In both cases, a pointer would be returned for the models stored in the model store.
        */
        auto start_time_selection = std::chrono::high_resolution_clock::now();
        std::map<std::string, std::vector<const Model *>> selected_models =
            model_store_->SelectModels(to_select_block);
        auto end_time_selection = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time_selection =
            end_time_selection - start_time_selection;
        auto avg_time_selection_per_model = elapsed_time_selection.count() / block_size;
        for (auto const &[selected_learner_id, selected_learner_models]: selected_models) {
          (*metadata_.at(metadata_ref_idx).mutable_model_selection_duration_ms())[selected_learner_id] =
              avg_time_selection_per_model;
        }

        /* --- CONSTRUCT MODELS TO AGGREGATE --- */
        for (auto const &[selected_learner_id, selected_learner_models]: selected_models) {
          auto scaling_factor = scaling_factors[selected_learner_id];
          for (auto it: selected_learner_models) {
            to_aggregate_learner_models_tmp.emplace_back(it, scaling_factor);
          }
          to_aggregate_block.push_back(to_aggregate_learner_models_tmp);
          to_aggregate_learner_models_tmp.clear();
        }

        /* --- AGGREGATE MODELS --- */
        auto start_time_block_aggregation = std::chrono::high_resolution_clock::now();
        new_community_model = aggregator_->Aggregate(to_aggregate_block);
        auto end_time_block_aggregation = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time_block_aggregation =
            end_time_block_aggregation - start_time_block_aggregation;
        *metadata_.at(metadata_ref_idx).mutable_model_aggregation_block_duration_ms()->Add() =
            elapsed_time_block_aggregation.count();

        long block_memory = GetTotalMemory();
        PLOG(INFO) << "Aggregate block memory usage (kb): " << block_memory;
        *metadata_.at(metadata_ref_idx).mutable_model_aggregation_block_memory_kb()->Add() = 
            (double) block_memory;

        // Cleanup. Clear sentinel block variables and reset
        // model_store's state to reclaim unused memory.
        to_select_block.clear();
        to_aggregate_block.clear();
        model_store_->ResetState();

      } // end-if

    } // end for loop

    // Reset aggregation function's state for the next step.
    aggregator_->Reset();

    // Compute elapsed time for the entire aggregation - global model computation function.
    auto end_time_aggregation = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_aggregation =
        end_time_aggregation - start_time_aggregation;
    metadata_.at(metadata_ref_idx).set_model_aggregation_total_duration_ms(elapsed_time_aggregation.count());
    *metadata_.at(metadata_ref_idx).mutable_model_aggregation_completed_at() = TimeUtil::GetCurrentTime();

    return new_community_model;

  }

  void RecordCommunityModelSize(const FederatedModel &model,
                                const uint32_t &metadata_ref_idx) {
    /*
     * Here, we record all tensor metadat associated with its size, non-zero and zero values.
     */
    for (auto &variable: model.model().variables()) {
      if (variable.has_plaintext_tensor()) {
        auto data_type = variable.plaintext_tensor().tensor_spec().type().type();
        auto tensor_spec = variable.plaintext_tensor().tensor_spec();
        TensorQuantifier tensor_quantifier;
        if (data_type == DType_Type_UINT8) {
          tensor_quantifier = ::proto::QuantifyTensor<unsigned char>(tensor_spec);
        } else if (data_type == DType_Type_UINT16) {
          tensor_quantifier = ::proto::QuantifyTensor<unsigned short>(tensor_spec);
        } else if (data_type == DType_Type_UINT32) {
          tensor_quantifier = ::proto::QuantifyTensor<unsigned int>(tensor_spec);
        } else if (data_type == DType_Type_UINT64) {
          tensor_quantifier = ::proto::QuantifyTensor<unsigned long>(tensor_spec);
        } else if (data_type == DType_Type_INT8) {
          tensor_quantifier = ::proto::QuantifyTensor<signed char>(tensor_spec);
        } else if (data_type == DType_Type_INT16) {
          tensor_quantifier = ::proto::QuantifyTensor<signed short>(tensor_spec);
        } else if (data_type == DType_Type_INT32) {
          tensor_quantifier = ::proto::QuantifyTensor<signed int>(tensor_spec);
        } else if (data_type == DType_Type_INT64) {
          tensor_quantifier = ::proto::QuantifyTensor<signed long>(tensor_spec);
        } else if (data_type == DType_Type_FLOAT32) {
          tensor_quantifier = ::proto::QuantifyTensor<float>(tensor_spec);
        } else if (data_type == DType_Type_FLOAT64) {
          tensor_quantifier = ::proto::QuantifyTensor<double>(tensor_spec);
        } else {
          throw std::runtime_error("Unsupported tensor data type.");
        } // end if

        // Record the computed tensor measurements.
        *metadata_.at(metadata_ref_idx).mutable_model_tensor_quantifiers()->Add()
            = tensor_quantifier;

      } else if (variable.has_ciphertext_tensor()) {
        auto tensor_quantifier = metisfl::TensorQuantifier();
        // Since the controller performs the aggregation in an encrypted space,
        // it does not have access to the plaintext model and therefore we cannot
        // find the number of zero and non-zero elements.
        tensor_quantifier.set_tensor_size_bytes(variable.ciphertext_tensor().tensor_spec().ByteSizeLong());
        *metadata_.at(metadata_ref_idx).mutable_model_tensor_quantifiers()->Add() =
            tensor_quantifier;
      } else {
        throw std::runtime_error("Unsupported variable tensor type.");
      } // end if

    } // end for

  }

  // Controllers parameters.
  ControllerParams params_;
  uint32_t global_iteration_;
  // We store a collection of federated training metadata as training
  // progresses related to the federation runtime environment. All
  // insertions take place at the end of the structure, and we want to
  // randomly access positions in the structure. Hence, the vector container.
  std::vector<FederatedTaskRuntimeMetadata> metadata_;
  // Stores learners' execution state inside a lookup map.
  absl::flat_hash_map<std::string, LearnerState> learners_;
  // Stores learners' connection stub.
  absl::flat_hash_map<std::string, LearnerStub> learners_stub_;
  absl::flat_hash_map<std::string, LearningTaskTemplate>
      learners_task_template_;
  // Stores local models evaluation lineages.
  absl::flat_hash_map<std::string, std::list<TaskExecutionMetadata>>
      local_tasks_metadata_;
  std::mutex learners_mutex_;
  // Stores community models evaluation lineages. A community model might not
  // get evaluated across all learners depending on the participation ratio and
  // therefore we store sequentially the evaluations on every other learner.
  // Insertions occur at the head of the structure, hence the use of list.
  std::vector<CommunityModelEvaluation> community_evaluations_;
  // Scaling function for computing the scaling factor of each learner.
  std::unique_ptr<ScalingFunction> scaler_;
  // Aggregation function to use for computing the community model.
  std::unique_ptr<AggregationFunction> aggregator_;
  // Federated task scheduler.
  std::unique_ptr<Scheduler> scheduler_;
  // Federated model selector.
  std::unique_ptr<Selector> selector_;
  // Community model.
  FederatedModel community_model_;
  // Thread pool for scheduling tasks.
  BS::thread_pool scheduling_pool_;
  // Caching function to use for storing learner model(s).
  std::unique_ptr<ModelStore> model_store_;
  std::mutex model_store_mutex_;
  // GRPC completion queue to process submitted learners' RunTasks requests.
  grpc::CompletionQueue run_tasks_cq_;
  // GRPC completion queue to process submitted learners' EvaluateModel requests.
  grpc::CompletionQueue eval_tasks_cq_;

  // Templated struct for keeping state and data information
  // from requests submitted to learners services.
  template<typename T>
  struct AsyncLearnerCall {

    std::string learner_id;

    // Container for the data we expect from the server.
    T reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    grpc::ClientContext context;

    // Storage for the status of the RPC upon completion.
    grpc::Status status;

    std::unique_ptr<grpc::ClientAsyncResponseReader<T>> response_reader;

  };

  // Implementation of generic AsyncLearnerCall type to handle RunTask responses.
  struct AsyncLearnerRunTaskCall : AsyncLearnerCall<RunTaskResponse> {};

  // Implementation of generic AsyncLearnerCall type to handle EvaluateModel responses.
  struct AsyncLearnerEvalCall : AsyncLearnerCall<EvaluateModelResponse> {
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

} // namespace

std::unique_ptr<Controller> Controller::New(const ControllerParams &params) {

  // Validate parameters correctness prior to Controller initialization.
  if (params.model_hyperparams().batch_size() == 0 ||
      params.model_hyperparams().epochs() == 0) {
    // We need both the batch size and the epochs to compute and assign the
    // initial number of steps learners will perform when joining the federation.
    throw std::runtime_error("Batch size and epochs cannot be zero.");
  }

  return absl::make_unique<ControllerDefaultImpl>(
      ControllerParams(params),
      CreateScaler(params.global_model_specs().aggregation_rule().aggregation_rule_specs()),
      CreateAggregator(params.global_model_specs().aggregation_rule()),
      CreateScheduler(params.communication_specs()),
      CreateSelector(),
      CreateModelStore(params.model_store_config()));
}

} // namespace metisfl::controller
