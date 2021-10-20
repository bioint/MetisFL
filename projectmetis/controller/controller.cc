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

#include <iostream>
#include <utility>

#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>

#include "absl/memory/memory.h"
#include "projectmetis/controller/model_aggregation/aggregations.h"
#include "projectmetis/controller/model_scaling/scalings.h"
#include "projectmetis/controller/scheduling/scheduling.h"
#include "projectmetis/controller/controller_utils.h"
#include "projectmetis/proto/learner.grpc.pb.h"
#include "projectmetis/proto/metis.pb.h"
#include "projectmetis/core/macros.h"

namespace projectmetis::controller {
namespace {

class ControllerDefaultImpl : public Controller {
 public:
  ControllerDefaultImpl(ControllerParams params,
                        std::unique_ptr<ScalingFunction> scaler,
                        std::unique_ptr<AggregationFunction> aggregator,
                        std::unique_ptr<Scheduler> scheduler)
      : params_(std::move(params)),
        scaler_(std::move(scaler)),
        aggregator_(std::move(aggregator)),
        scheduler_(std::move(scheduler)),
        community_model_() {}

  const ControllerParams &GetParams() const override {
    return params_;
  }

  std::vector<LearnerDescriptor> GetLearners() const override {
    std::vector<LearnerDescriptor> learners;
    for (const auto &[key, learner_state] : learners_) {
      learners.push_back(learner_state.learner());
    }
    return learners;
  }

  uint32_t GetNumLearners() const override {
    return learners_.size();
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
      return absl::InvalidArgumentError("Invalid dataset spec provided.");
    }

    // TODO(dstripelis) Condition to ping the hostname + port.

    // Generates learner id.
    const std::string learner_id = GenerateLearnerId(server_entity);

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
    learner_stubs_[learner_id] = Learner::NewStub(channel);

    return learner;
  }

  absl::Status RemoveLearner(const std::string &learner_id,
                             const std::string &token) override {
    RETURN_IF_ERROR(ValidateLearner(learner_id, token));

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

  absl::Status LearnerCompletedTask(const std::string &learner_id,
                                    const std::string &token,
                                    const CompletedLearningTask &task) override {
    RETURN_IF_ERROR(ValidateLearner(learner_id, token));

    // Updates the learner's state.
    *learners_[learner_id].mutable_model()->Add() = task.model();

    auto
        to_schedule = scheduler_->ScheduleNext(learner_id, task, GetLearners());
    if (!to_schedule.empty()) {
      auto scaling_factors =
          scaler_->ComputeScalingFactors(community_model_, learners_);

      std::vector<std::pair<const Model *, double>> participating_models;
      for (const auto&[learner_id, state]: learners_) {
        const auto history_size = state.model_size();
        const auto latest_model = state.model(history_size - 1);
        const auto scaling_factor = scaling_factors[learner_id];

        participating_models.emplace_back(std::make_pair(&latest_model,
                                                         scaling_factor));
      }

      auto community_model = aggregator_->Aggregate(participating_models);

      for (const auto &to_schedule_id : to_schedule) {
        ::grpc::ClientContext context;
        RunTaskResponse response;
        RunTaskRequest request;
        *request.mutable_federated_model() = community_model;
        // TODO: define learning task. how many steps?

        // Finds learner gRPC channel and sends RunTask request.
        learner_stubs_[to_schedule_id]->RunTask(&context, request, &response);

        // TODO: check if response indicates a success.
      }

      // Updates the community model.
      // TODO: maybe we want to keep track of the lineage?
      community_model_ = community_model;
    }

    return absl::OkStatus();
  }

 private:
  typedef std::unique_ptr<Learner::Stub> LearnerStub;

  absl::Status ValidateLearner(const std::string &learner_id,
                               const std::string &token) const {
    // Validates non-empty learner_id and authentication token.
    if (learner_id.empty() || token.empty()) {
      return absl::InvalidArgumentError("Learner id and token cannot be empty");
    }

    const auto &learner = learners_.find(learner_id);
    if (learner == learners_.end()
        || learner->second.learner().auth_token() != token) {
      return absl::PermissionDeniedError("Invalid token provided.");
    }

    return absl::OkStatus();
  }

  // Controllers parameters.
  ControllerParams params_;
  // Stores active learners execution state inside a lookup map.
  absl::flat_hash_map<std::string, LearnerState> learners_;
  absl::flat_hash_map<std::string, LearnerStub> learner_stubs_;
  // Scaling function for computing the scaling factor of each learner.
  std::unique_ptr<ScalingFunction> scaler_;
  // Aggregation function to use for computing the community model.
  std::unique_ptr<AggregationFunction> aggregator_;
  // Federated task scheduler.
  std::unique_ptr<Scheduler> scheduler_;
  // Community model.
  FederatedModel community_model_;
};

std::unique_ptr<AggregationFunction> CreateAggregator(const GlobalModelSpecs &specs) {
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
  return absl::make_unique<ControllerDefaultImpl>(params,
                                                  absl::make_unique<
                                                      DatasetSizeScaler>(),
                                                  CreateAggregator(params.global_model_specs()),
                                                  CreateScheduler(params.communication_specs()));
}

} // namespace projectmetis::controller
