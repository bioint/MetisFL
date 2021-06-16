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

#include "projectmetis/controller/controller_servicer.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "projectmetis/proto/controller.grpc.pb.h"
#include "projectmetis/proto/shared.grpc.pb.h"

namespace projectmetis::controller {
namespace {
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;

class ControllerServicerImpl : public ControllerServicer {
public:
  explicit ControllerServicerImpl(Controller *controller)
      : controller_(controller) {
    GOOGLE_CHECK_NOTNULL(controller_);
  }

  ABSL_MUST_USE_RESULT
  const Controller *GetController() const override { return controller_; }

  Status GetParticipatingLearners(
      ServerContext *context, const GetParticipatingLearnersRequest *request,
      GetParticipatingLearnersResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Request and response cannot be empty.");
    }

    // Creates LearnerEntity response collection.
    for (const auto &learner : controller_->GetLearners()) {
      std::cout << learner.DebugString() << std::endl;
      *response->add_server_entity() = learner.server_entity();
    }

    return Status::OK;
  }

  Status JoinFederation(ServerContext *context,
                        const JoinFederationRequest *request,
                        JoinFederationResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Request and response cannot be empty.");
    }

    // Validates that the incoming request has the required fields populated.
    if (!request->has_server_entity() && !request->has_local_dataset_spec()) {
      response->mutable_ack()->set_status(false);
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Server entity and local dataset cannot be empty.");
    }

    const auto learner_state_or = controller_->AddLearner(
        request->server_entity(), request->local_dataset_spec());

    if (!learner_state_or.ok()) {
      response->mutable_ack()->set_status(false);
      // Returns the internal status error message as servicer's status message.
      return Status(StatusCode::ALREADY_EXISTS,
                    std::string(learner_state_or.status().message()));
    } else {
      response->mutable_ack()->set_status(true);

      const auto &learner_state = learner_state_or.value();
      response->set_learner_id(learner_state.learner_id());
      response->set_auth_token(learner_state.auth_token());
    }

    return Status::OK;
  }

  Status LeaveFederation(ServerContext *context,
                         const LeaveFederationRequest *request,
                         LeaveFederationResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Request and response cannot be empty.");
    }

    // Validates that the incoming request has the required fields populated.
    if (request->learner_id().empty() || request->auth_token().empty()) {
      response->mutable_ack()->set_status(false);
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Learner id and authentication token cannot be empty.");
    }

    const std::string &learner_id = request->learner_id();
    const std::string &auth_token = request->auth_token();

    const auto del_status = controller_->RemoveLearner(learner_id, auth_token);

    if (del_status.ok()) {
      response->mutable_ack()->set_status(true);
      return Status::OK;
    } else {
      response->mutable_ack()->set_status(false);
      return Status(StatusCode::CANCELLED, std::string(del_status.message()));
    }
  }

private:
  Controller *controller_;
};
} // namespace

std::unique_ptr<ControllerServicer>
ControllerServicer::New(Controller *controller) {
  return absl::make_unique<ControllerServicerImpl>(controller);
}

} // namespace projectmetis::controller
