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

#include <memory>
#include <utility>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "absl/memory/memory.h"
#include "projectmetis/controller/controller_servicer.h"
#include "projectmetis/proto/controller.grpc.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {
namespace {
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;

class ServicerBase {
 public:

  template<class Service>
  void Start(const std::string& hostname, uint32_t port, Service* service) {
    const auto server_address = absl::StrCat(hostname, ":", port);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;

    // Listens on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Registers "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(service);

    // Finally assemble the server.
    server_ = builder.BuildAndStart();
    std::cout << "Controller listening on " << server_address << std::endl;
  }

  void Stop() {
    if (server_ == nullptr) {
      return;
    }

    server_->Shutdown();
    server_ = nullptr;
    std::cout << "Controller has shut down" << std::endl;
  }

  void Wait() {
    if (server_ == nullptr) {
      return;
    }

    server_->Wait();
  }

 protected:
  std::unique_ptr<Server> server_;
};

class ControllerServicerImpl : public ControllerServicer, private ServicerBase {
 public:
  explicit ControllerServicerImpl(Controller *controller)
      : controller_(controller) {
    GOOGLE_CHECK_NOTNULL(controller_);
  }

  ABSL_MUST_USE_RESULT
  const Controller *GetController() const override { return controller_; }

  void StartService() override {
    const auto& params = controller_->GetParams();

    Start(params.server_entity().hostname(), params.server_entity().port(), this);
  }

  void WaitService() override {
    Wait();
  }

  void StopService() override {
    Stop();
  }

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
      *response->add_server_entity() = learner.service_spec();
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

    const auto learner_or = controller_->AddLearner(
        request->server_entity(), request->local_dataset_spec());

    if (!learner_or.ok()) {
      response->mutable_ack()->set_status(false);
      // Returns the internal status error message as servicer's status message.
      return Status(StatusCode::ALREADY_EXISTS,
                    std::string(learner_or.status().message()));
    } else {
      response->mutable_ack()->set_status(true);

      const auto &learner_state = learner_or.value();
      response->set_learner_id(learner_state.id());
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
