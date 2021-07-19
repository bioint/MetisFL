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

#include <iostream>
#include <utility>

#include "absl/memory/memory.h"
#include "projectmetis/controller/controller.h"
#include "projectmetis/controller/controller_utils.h"
#include "projectmetis/controller/model_aggregation/aggregations.h"
#include "projectmetis/proto/controller.pb.h"

namespace projectmetis::controller {
namespace {

std::unique_ptr<AggregationFunction>
CreateAggregator(GlobalModelSpecs::AggregationRule rule) {
  // TODO (dstripelis) Create a util function to fetch the supporting model
  //  aggregation rules - needs to also be invoked with Python Bindings.
  if (rule == GlobalModelSpecs::FED_AVG) {
    return absl::make_unique<FederatedAverage>();
  }

  throw std::runtime_error("Unsupported aggregation rule.");
}

class ControllerDefaultImpl : public Controller {
 public:
  // std::move() clears the controller_params collection,
  // hence we need to access the initialized private collection.
  explicit ControllerDefaultImpl(ControllerParams controller_params)
      : params_(std::move(controller_params)),
        aggregator_(CreateAggregator(
            params_.global_model_specs().aggregation_rule())) {}

  const ControllerParams &GetParams() const override {
    return params_;
  }

  std::vector<LearnerState> GetLearners() const override {
    std::vector<LearnerState> learners;
    for (const auto &[key, learner] : learners_) {
      learners.push_back(learner);
    }
    return learners;
  }

  absl::StatusOr<LearnerState>
  AddLearner(const ServerEntity &server_entity,
             const DatasetSpec &dataset_spec) override {
    // Validates non-empty hostname and non-negative port.
    if (server_entity.hostname().empty() || server_entity.port() < 0) {
      return absl::InvalidArgumentError("Hostname and port must be provided.");
    }

    // Validates number of train, validation and test examples. Train examples
    // must always be positive, while validation and test can be non-negative.
    if (dataset_spec.num_training_examples() <= 0 ||
        dataset_spec.num_validation_examples() < 0 ||
        dataset_spec.num_test_examples() < 0) {
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
    LearnerState learner_state;
    learner_state.set_learner_id(learner_id);
    *learner_state.mutable_server_entity() = server_entity;
    learner_state.set_auth_token(auth_token);
    *learner_state.mutable_local_dataset_spec() = dataset_spec;

    // Registers learner.
    learners_[learner_id] = learner_state;

    return learner_state;
  }

  absl::Status RemoveLearner(const std::string &learner_id,
                             const std::string &token) override {

    // Validates non-empty learner_id and authentication token.
    if (learner_id.empty() || token.empty()) {
      return absl::InvalidArgumentError("Learner id and token cannot be empty");
    }

    auto it = learners_.find(learner_id);
    // Checks requesting learner existence inside the state map.
    if (it != learners_.end()) {
      if (it->second.auth_token() == token) {
        learners_.erase(it);
        return absl::OkStatus();
      } else {
        return absl::UnauthenticatedError("Learner token is wrong.");
      }
    } else {
      return absl::NotFoundError("Learner is not part of the federation.");
    }
  }

 private:
  // Controllers parameters.
  ControllerParams params_;
  // Stores active learners execution state inside a lookup map.
  absl::flat_hash_map<std::string, LearnerState> learners_;
  // Aggregation function to use for computing the community model.
  std::unique_ptr<AggregationFunction> aggregator_;
};

} // namespace

std::unique_ptr<Controller> Controller::New(const ControllerParams &params) {
  return absl::make_unique<ControllerDefaultImpl>(params);
}

} // namespace projectmetis::controller
