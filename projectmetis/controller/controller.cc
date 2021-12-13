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

#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include "absl/memory/memory.h"
#include "projectmetis/controller/controller_utils.h"
#include "projectmetis/controller/model_aggregation/aggregations.h"
#include "projectmetis/controller/model_scaling/scalings.h"
#include "projectmetis/controller/scheduling/scheduling.h"
#include "projectmetis/core/macros.h"
#include "projectmetis/core/thread_pool.h"
#include "projectmetis/proto/learner.grpc.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {
namespace {

class ControllerDefaultImpl : public Controller {
public:
  ControllerDefaultImpl(ControllerParams params,
                        std::unique_ptr<ScalingFunction> scaler,
                        std::unique_ptr<AggregationFunction> aggregator,
                        std::unique_ptr<Scheduler> scheduler)
      : params_(std::move(params)), global_iteration_(0), learners_(),
        learner_stubs_(), learners_mutex_(), scaler_(std::move(scaler)),
        aggregator_(std::move(aggregator)), scheduler_(std::move(scheduler)),
        // TODO(canastas) needed to increase thread pool count to carry out
        //  global iterations. Should the count be equal to the total number of
        //  participating learners?
        community_model_(), community_mutex_(), pool_(10) {}

  const ControllerParams &GetParams() const override { return params_; }

  std::vector<LearnerDescriptor> GetLearners() const override {
    std::vector<LearnerDescriptor> learners;
    for (const auto &[key, learner_state] : learners_) {
      learners.push_back(learner_state.learner());
    }
    return learners;
  }

  uint32_t GetNumLearners() const override { return learners_.size(); }

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
      return absl::InvalidArgumentError("Invalid dataset spec provided.");
    }

    // TODO(dstripelis) Condition to ping the hostname + port.

    // Generates learner id.
    const std::string learner_id = GenerateLearnerId(server_entity);

    // Acquires a lock to avoid having multiple threads overwriting the learners
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

    // Registers learner.
    learners_[learner_id] = learner_state;

    // Opens gRPC channel with learner.
    auto target =
        absl::StrCat(server_entity.hostname(), ":", server_entity.port());
    auto channel =
        ::grpc::CreateChannel(target, ::grpc::InsecureChannelCredentials());
    learner_stubs_[learner_id] = LearnerService::NewStub(channel);

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

    // Updates the learner's state.
    *learners_[learner_id].mutable_model()->Add() = task.model();

    // Schedules next tasks if necessary.
    ScheduleTasks(learner_id, task);

    return absl::OkStatus();
  }

  std::vector<ModelEvaluation>
  GetEvaluationLineage(const std::string &learner_id,
                       uint32_t num_steps) override {
    if (!evaluations_.contains(learner_id)) {
      return {};
    }

    const auto &lineage = evaluations_[learner_id];

    if (num_steps <= 0) {
      return {lineage.begin(), lineage.end()};
    }

    std::vector<ModelEvaluation> lineage_head;
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

  void ScheduleTasks(const std::string &learner_id,
                     const CompletedLearningTask &task) {
    // Acquires a lock to avoid having multiple threads overwriting the
    // community model. The guard releases the mutex as soon as it goes out of
    // scope so no need to manually release it in the code.
    std::lock_guard<std::mutex> guard(community_mutex_);

    auto to_schedule =
        scheduler_->ScheduleNext(learner_id, task, GetLearners());
    if (!to_schedule.empty()) {
      // Increases global step.
      ++global_iteration_;
      std::cout << "Global Iteration: " << unsigned(global_iteration_) << std::endl;

      auto community_model = ComputeCommunityModel();
      community_model.set_global_iteration(global_iteration_);

      for (const auto &to_schedule_id : to_schedule) {
        const auto &learner = learners_[to_schedule_id].learner();
        auto *learner_stub = learner_stubs_[to_schedule_id].get();
        const auto &dataset_spec = learner.dataset_spec();
        auto &params = params_;
        // Send evaluation tasks.
        if (!evaluations_.contains(to_schedule_id)) {
          evaluations_[to_schedule_id] = std::list<ModelEvaluation>();
        }
        auto &evaluation_lineage = evaluations_[to_schedule_id];
        pool_.push_task([learner_stub, params, community_model, &evaluation_lineage] {
          EvaluateModelRequest request;
          *request.mutable_model() = community_model.model();
          request.set_batch_size(params.model_hyperparams().batch_size());
          request.add_evaluation_dataset(EvaluateModelRequest::TRAINING);
          request.add_evaluation_dataset(EvaluateModelRequest::VALIDATION);
          request.add_evaluation_dataset(EvaluateModelRequest::TEST);
          // TODO(stripeli): metrics??

          ::grpc::ClientContext context;
          EvaluateModelResponse response;
          auto status =
              learner_stub->EvaluateModel(&context, request, &response);

          if (status.ok()) {
            evaluation_lineage.push_front(response.evaluation());
          }
        });

        // Send next training tasks.
        pool_.push_task([learner_stub, dataset_spec, community_model, &params] {
          RunTaskRequest request;
          *request.mutable_federated_model() = community_model;

          auto *next_task = request.mutable_task();
          const auto &model_params = params.model_hyperparams();
          uint32_t steps_per_epoch =
              dataset_spec.num_training_examples() / model_params.batch_size();
          next_task->set_num_local_updates(model_params.epochs() *
                                           steps_per_epoch);
          next_task->set_training_dataset_percentage_for_stratified_validation(
              model_params.percent_validation());

          auto *hyperparams = request.mutable_hyperparameters();
          hyperparams->set_batch_size(model_params.batch_size());
          *hyperparams->mutable_optimizer() =
              params.model_hyperparams().optimizer();

          ::grpc::ClientContext context;
          RunTaskResponse response;
          learner_stub->RunTask(&context, request, &response);
        });
      }

      // Updates the community model.
      community_model_ = community_model;
    }
  }

  FederatedModel ComputeCommunityModel() {
    auto scaling_factors =
        scaler_->ComputeScalingFactors(community_model_, learners_);
    std::vector<std::pair<const Model*, double>> participating_models;
    for (const auto &[id, state] : learners_) {
      const auto history_size = state.model_size();
      const auto& latest_model = state.model(history_size - 1);
      const auto scaling_factor = scaling_factors[id];
      participating_models.emplace_back(
          std::make_pair(&latest_model, scaling_factor));
    }
    return aggregator_->Aggregate(participating_models);
  }

  // Controllers parameters.
  ControllerParams params_;
  uint8_t global_iteration_;
  // Stores active learners execution state inside a lookup map.
  absl::flat_hash_map<std::string, LearnerState> learners_;
  absl::flat_hash_map<std::string, LearnerStub> learner_stubs_;
  std::mutex learners_mutex_;
  // Stores evaluation lineages
  absl::flat_hash_map<std::string, std::list<ModelEvaluation>> evaluations_;
  // Scaling function for computing the scaling factor of each learner.
  std::unique_ptr<ScalingFunction> scaler_;
  // Aggregation function to use for computing the community model.
  std::unique_ptr<AggregationFunction> aggregator_;
  // Federated task scheduler.
  std::unique_ptr<Scheduler> scheduler_;
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
  if (specs.protocol() == CommunicationSpecs::SYNCHRONOUS) {
    return absl::make_unique<SynchronousScheduler>();
  }
  throw std::runtime_error("unsupported scheduling policy.");
}

} // namespace

std::unique_ptr<Controller> Controller::New(const ControllerParams &params) {
  return absl::make_unique<ControllerDefaultImpl>(
      params, absl::make_unique<DatasetSizeScaler>(),
      CreateAggregator(params.global_model_specs()),
      CreateScheduler(params.communication_specs()));
}

} // namespace projectmetis::controller
