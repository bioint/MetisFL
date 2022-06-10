// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "projectmetis/controller/controller.h"

#include <mutex>
#include <utility>

#include <google/protobuf/util/time_util.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include "absl/memory/memory.h"
#include "projectmetis/controller/controller_utils.h"
#include "projectmetis/controller/model_aggregation/model_aggregation.h"
#include "projectmetis/controller/model_scaling/model_scaling.h"
#include "projectmetis/controller/model_selection/model_selection.h"
#include "projectmetis/controller/scheduling/scheduling.h"
#include "projectmetis/core/macros.h"
#include "projectmetis/core/thread_pool.h"
#include "projectmetis/proto/learner.grpc.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {
namespace {

using google::protobuf::util::TimeUtil;

class ControllerDefaultImpl : public Controller {
public:
  ControllerDefaultImpl(ControllerParams &&params,
                        std::unique_ptr<ScalingFunction> scaler,
                        std::unique_ptr<AggregationFunction> aggregator,
                        std::unique_ptr<Scheduler> scheduler,
                        std::unique_ptr<Selector> selector)
      : params_(std::move(params)), global_iteration_(0), learners_(),
        learner_stubs_(), learner_task_templates_(), learners_mutex_(),
        scaler_(std::move(scaler)), aggregator_(std::move(aggregator)),
        scheduler_(std::move(scheduler)), selector_(std::move(selector)),
        community_model_(), community_mutex_(), pool_(2) {}

  const ControllerParams &GetParams() const override { return params_; }

  std::vector<LearnerDescriptor> GetLearners() const override {
    // TODO Shall we 'hide' authentication token from exposure?
    std::vector<LearnerDescriptor> learners;
    for (const auto &[key, learner_state] : learners_) {
      learners.push_back(learner_state.learner());
    }
    return learners;
  }

  uint32_t GetNumLearners() const override { return learners_.size(); }

  const FederatedModel &CommunityModel() const override {
    return community_model_;
  }

  // TODO: add admin auth token support
  absl::Status ReplaceCommunityModel(const FederatedModel &model) override {
    std::lock_guard<std::mutex> guard(community_mutex_);
    // When replacing the federation model we only set the number of learners
    // contributed to this model and the actual model. We do not replace the
    // global iteration since it is updated exclusively by the controller.
    community_model_.set_num_contributors(model.num_contributors());
    *community_model_.mutable_model() = model.model();
    return absl::OkStatus();
  }

  absl::StatusOr<LearnerDescriptor>
  AddLearner(const ServerEntity &server_entity,
             const DatasetSpec &dataset_spec) override {
    // Validates non-empty hostname and non-negative port.
    if (server_entity.hostname().empty() || server_entity.port() < 0) {
      return absl::InvalidArgumentError("Hostname and port must be provided.");
    }

    // Validates number of train, validation and test examples. Train examples
    // must always be positive, while validation and test can be non-negative.
    if (dataset_spec.num_training_examples() <= 0) {
      return absl::InvalidArgumentError("Learner training examples <= 0.");
    }

    // TODO(dstripelis) Condition to ping the hostname + port.

    // Generates learner id.
    const std::string learner_id = GenerateLearnerId(server_entity);

    // Acquires a lock to avoid having multiple threads overwriting the learners'
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> guard(learners_mutex_);

    if (learners_.contains(learner_id)) {
      // Learner was already registered with the controller.
      return absl::AlreadyExistsError("Learner has already joined.");
    }

    // Generates an auth token for the learner.
    // TODO(canastas) We need a better authorization token generator.
    const std::string auth_token = std::to_string(learners_.size() + 1);

    // Initializes learner state with an empty model.
    LearnerDescriptor learner;
    learner.set_id(learner_id);
    learner.set_auth_token(auth_token);
    *learner.mutable_service_spec() = server_entity;
    *learner.mutable_dataset_spec() = dataset_spec;

    LearnerState learner_state;
    *learner_state.mutable_learner() = learner;

    // Creates default task template.
    LearningTaskTemplate task_template;
    uint32_t steps_per_epoch = dataset_spec.num_training_examples() /
                               params_.model_hyperparams().batch_size();
    task_template.set_num_local_updates(params_.model_hyperparams().epochs() *
                                        steps_per_epoch);

    // Registers learner.
    learners_[learner_id] = learner_state;
    learner_task_templates_[learner_id] = task_template;

    // Opens gRPC channel with learner.
    auto target =
        absl::StrCat(server_entity.hostname(), ":", server_entity.port());
    auto channel =
        ::grpc::CreateChannel(target, ::grpc::InsecureChannelCredentials());
    learner_stubs_[learner_id] = LearnerService::NewStub(channel);

    // Triggers the initial task.
    ScheduleInitialTask(learner_id);

    return learner;
  }

  absl::Status RemoveLearner(const std::string &learner_id,
                             const std::string &token) override {
    RETURN_IF_ERROR(ValidateLearner(learner_id, token));

    // Acquires a lock to avoid having multiple threads overwriting the learners
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> guard(learners_mutex_);

    auto it = learners_.find(learner_id);
    // Checks requesting learner existence inside the state map.
    if (it != learners_.end()) {
      if (it->second.learner().auth_token() == token) {
        learners_.erase(it);
        learner_task_templates_.erase(learner_id);
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

    // Assign a non-negative value to the metadata index.
    auto task_global_iteration = task.execution_metadata().global_iteration();
    auto metadata_index =
        task_global_iteration == 0 ? 0 : task_global_iteration - 1;
    // Records the id of the learner completed the task.
    if (not metadata_.empty() && metadata_index < metadata_.size()) {
      *metadata_.at(metadata_index).add_completed_by_learner_id() = learner_id;
    }

    // Updates the learner's state.
    *learners_[learner_id].mutable_model()->Add() = task.model();

    // Update learner collection with metrics from last completed training task.
    if (!local_tasks_metadata_.contains(learner_id)) {
      local_tasks_metadata_[learner_id] = std::list<TaskExecutionMetadata>();
    }
    local_tasks_metadata_[learner_id].push_front(task.execution_metadata());

    // Schedules next tasks if necessary. We call ScheduleTasks() asynchronously
    // because during synchronous execution, the learner who completed its local
    // training task the last within a federation round will have to wait for
    // all the rest of training tasks to be scheduled before it can receive an
    // acknowledgement by the controller for its completed task. Put it simply,
    // the learner who completed its task the last within a round will have to
    // keep a connection open with the controller, till the controller schedules
    // all necessary training tasks for the next federation round.
//    pool_.push_task(
//        [this, learner_id, task] { ScheduleTasks(learner_id, task); });
    ScheduleTasks(learner_id, task);

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

private:
  typedef std::unique_ptr<LearnerService::Stub> LearnerStub;

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

    if (metadata_.empty()) {
      // When the very first local training task is scheduled, we need to
      // increase the global iteration counter and create the first
      // federation runtime metadata object.
      FederatedTaskRuntimeMetadata meta = FederatedTaskRuntimeMetadata();
      ++global_iteration_;
      std::cout << "FedIteration: " << unsigned(global_iteration_) << std::endl;
      meta.set_global_iteration(global_iteration_);
      *meta.mutable_started_at() = TimeUtil::GetCurrentTime();
      metadata_.emplace_back(meta);
    }

    // Records the learner id to which the controller delegates the latest task.
    *metadata_.back().add_assigned_to_learner_id() = learner_id;
    auto &community_model = community_model_;

    // Send initial training.
    SendRunTask(learner_id, community_model);
  }

  void ScheduleTasks(const std::string &learner_id,
                     const CompletedLearningTask &task) {
    // Acquires a lock to avoid having multiple threads overwriting the learners
    // data structures. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> guard(learners_mutex_);
    // TODO Maybe for global_iteration_ as well?

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
      auto to_select = selector_->Select(to_schedule, GetLearners());
      // Computes the community model using models that have
      // been selected by the model selector.
      auto community_model = ComputeCommunityModel(to_select);
      community_model.set_global_iteration(task_global_iteration);
      // Updates the community model.
      community_model_ = community_model;

      // Creates an evaluation hash map container for the new community model.
      CommunityModelEvaluation community_eval = CommunityModelEvaluation();
      // Records the evaluation of the community model that was
      // computed at the previously completed global iteration.
      community_eval.set_global_iteration(task_global_iteration);
      community_evaluations_.emplace_back(community_eval);

      // Firstly, evaluate the community model across all scheduled learners.
      for (const auto &to_schedule_id : to_schedule) {
        // Send community model evaluation tasks to every learner. Each learner
        // holds a different training, validation and test dataset and hence
        // we need to evaluate the community model over each individual dataset.
        SendEvaluationTask(to_schedule_id, community_model);
      }
      // Wait for all evaluation tasks to complete
      // before starting a new global iteration.
      pool_.wait_for_tasks();

      // Increase global iteration counter to reflect the new scheduling round.
      ++global_iteration_;
      std::cout << "FedIteration: " << unsigned(global_iteration_) << std::endl;

      // Creates a new federation runtime metadata
      // object for the new scheduling round.
      FederatedTaskRuntimeMetadata meta = FederatedTaskRuntimeMetadata();
      meta.set_global_iteration(global_iteration_);
      *meta.mutable_started_at() = TimeUtil::GetCurrentTime();
      // Records the id of the learners to which
      // the controller delegates the training task.
      for (const auto &to_schedule_id : to_schedule) {
        *meta.add_assigned_to_learner_id() = to_schedule_id;
      }
      metadata_.emplace_back(meta);

      UpdateLearnersTaskTemplates(to_schedule);

      // Send training task to all scheduled learners.
      for (const auto &to_schedule_id : to_schedule) {
        // Send the new training task and add it to the pool.
        SendRunTask(to_schedule_id, community_model);
      }
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
    if (communication_specs.protocol() ==
        CommunicationSpecs::SEMI_SYNCHRONOUS &&
        (global_iteration_ == 2 ||
            protocol_specs.semi_sync_recompute_num_updates())) {
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
      float t_max = static_cast<float>(
          protocol_specs.semi_sync_lambda()) *
          ms_per_epoch_slowest;

      // Updates the task templates based on the slowest learner.
      for (const auto &learner_id : learners) {
        const auto &metadata = local_tasks_metadata_[learner_id].front();

        int num_local_updates =
            std::ceil(t_max / metadata.processing_ms_per_batch());

        auto &task_template = learner_task_templates_[learner_id];
        task_template.set_num_local_updates(num_local_updates);
      }
    }
  }

  void SendEvaluationTask(const std::string &learner_id,
                          const FederatedModel &model) {
    auto *learner_stub = learner_stubs_[learner_id].get();
    auto &params = params_;

    auto &model_evaluations_map = community_evaluations_.back();
    pool_.push_task([model, learner_stub, learner_id, &params,
                     &model_evaluations_map] {
      EvaluateModelRequest request;
      *request.mutable_model() = model.model();
      request.set_batch_size(params.model_hyperparams().batch_size());
      request.add_evaluation_dataset(EvaluateModelRequest::TRAINING);
      request.add_evaluation_dataset(EvaluateModelRequest::VALIDATION);
      request.add_evaluation_dataset(EvaluateModelRequest::TEST);

      ::grpc::ClientContext context;
      EvaluateModelResponse response;
      auto status = learner_stub->EvaluateModel(&context, request, &response);
      if (status.ok()) {
        ModelEvaluations model_evaluations;
        *model_evaluations.mutable_training_evaluation() =
            response.evaluations().training_evaluation();
        *model_evaluations.mutable_validation_evaluation() =
            response.evaluations().validation_evaluation();
        *model_evaluations.mutable_test_evaluation() =
            response.evaluations().test_evaluation();
        (*model_evaluations_map.mutable_evaluations())[learner_id] =
            model_evaluations;
      }
    });
  }

  void SendRunTask(const std::string &learner_id, const FederatedModel &model) {
    auto &params = params_;
    auto global_iteration = global_iteration_;
    const auto &learner = learners_[learner_id].learner();
    auto *learner_stub = learner_stubs_[learner_id].get();
    const auto &task_template = learner_task_templates_[learner_id];
    const auto &dataset_spec = learner.dataset_spec();

    pool_.push_task([learner_stub, task_template, dataset_spec, model,
                     global_iteration, &params] {
      RunTaskRequest request;
      *request.mutable_federated_model() = model;

      auto *next_task = request.mutable_task();
      next_task->set_global_iteration(global_iteration);
      const auto &model_params = params.model_hyperparams();
      next_task->set_num_local_updates(
          task_template.num_local_updates()); // get from task template.
      next_task->set_training_dataset_percentage_for_stratified_validation(
          model_params.percent_validation());
      // TODO (dstripelis) Add evaluation metrics for the learning task.

      auto *hyperparams = request.mutable_hyperparameters();
      hyperparams->set_batch_size(model_params.batch_size());
      *hyperparams->mutable_optimizer() =
          params.model_hyperparams().optimizer();

      ::grpc::ClientContext context;
      RunTaskResponse response;

      // TODO(aasghar) Need to implement logic, when the learner is behaving as
      // a server, and controller needs to connect.
      learner_stub->RunTask(&context, request, &response);
    });
  }

  FederatedModel
  ComputeCommunityModel(const std::vector<std::string> &learners_ids) {

    // Handles the case where the community model is requested for the
    // first time and has the original (random) initialization state.
    if (global_iteration_ < 2 && community_model_.IsInitialized()) {
      return community_model_;
    }

    // TODO (dstripelis) Remove redundant copying.
    absl::flat_hash_map<std::string, LearnerState> participating_states;
    for (const auto &id : learners_ids) {
      participating_states[id] = learners_.at(id);
    }
    auto scaling_factors =
        scaler_->ComputeScalingFactors(community_model_, participating_states);
    std::vector<std::pair<const Model *, double>> participating_models;
    for (const auto &[id, state] : participating_states) {
      if (not state.model().empty()) {
        const auto history_size = state.model_size();
        const auto &latest_model = state.model(history_size - 1);
        const auto scaling_factor = scaling_factors[id];
        participating_models.emplace_back(
            std::make_pair(&latest_model, scaling_factor));
      }
    }
    return aggregator_->Aggregate(participating_models);
  }

  // Controllers parameters.
  ControllerParams params_;
  uint32_t global_iteration_;
  // We store a collection of federated training metadata as training
  // progresses related to the federation runtime environment. All
  // insertions take place at the end of the structure and we want to
  // randomly access positions in the structure. Hence, the vector container.
  std::vector<FederatedTaskRuntimeMetadata> metadata_;
  // Stores active learners execution state inside a lookup map.
  absl::flat_hash_map<std::string, LearnerState> learners_;
  absl::flat_hash_map<std::string, LearnerStub> learner_stubs_;
  absl::flat_hash_map<std::string, LearningTaskTemplate>
      learner_task_templates_;
  std::mutex learners_mutex_;
  // Stores local models evaluation lineages.
  absl::flat_hash_map<std::string, std::list<TaskExecutionMetadata>>
      local_tasks_metadata_;
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
  std::mutex community_mutex_;
  // Thread pool for async tasks.
  thread_pool pool_;
};

std::unique_ptr<AggregationFunction>
CreateAggregator(const GlobalModelSpecs &specs) {
  if (specs.aggregation_rule() == GlobalModelSpecs::FED_AVG) {
    return absl::make_unique<FederatedAverage>();
  }
  throw std::runtime_error("unsupported aggregation rule.");
}

std::unique_ptr<Scheduler> CreateScheduler(const CommunicationSpecs &specs) {
  if (specs.protocol() == CommunicationSpecs::SYNCHRONOUS ||
      specs.protocol() == CommunicationSpecs::SEMI_SYNCHRONOUS) {
    return absl::make_unique<SynchronousScheduler>();
  }
  if (specs.protocol() == CommunicationSpecs::ASYNCHRONOUS) {
    return absl::make_unique<AsynchronousScheduler>();
  }
  throw std::runtime_error("unsupported scheduling policy.");
}

std::unique_ptr<Selector> CreateSelector() {
  return absl::make_unique<ScheduledCardinality>();
}

} // namespace

std::unique_ptr<Controller> Controller::New(const ControllerParams &params) {
  return absl::make_unique<ControllerDefaultImpl>(
      ControllerParams(params), absl::make_unique<DatasetSizeScaler>(),
      CreateAggregator(params.global_model_specs()),
      CreateScheduler(params.communication_specs()), CreateSelector());
}

} // namespace projectmetis::controller
