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
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;

class ServicerBase {
public:
  template <class Service>
  void Start(const ServerEntity &server_entity, Service *service) {
    const auto server_address = absl::StrCat(server_entity.hostname(), ":", server_entity.port());

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    std::shared_ptr<grpc::ServerCredentials> creds;

    bool read_ssl = false;

    /*! Read the server certificate & server key
     * and enable SSL accordingly. */
    if (!server_entity.ssl_config().server_cert().empty()
            && !server_entity.ssl_config().server_key().empty()) {
                read_ssl = true;
    }

    if (read_ssl) {
      std::string server_cert;
      std::string server_key;
      // TODO(aasghar) Understand how we can handle the SSL certificates if passed as commandline argument.
      std::string abs_path = std::filesystem::current_path();
      std::string temp_cert  = abs_path + server_entity.ssl_config().server_cert();
      std::string temp_key =  abs_path + server_entity.ssl_config().server_key();

      if (read_and_parse_file(server_cert, temp_cert) == -1) {
          std::cerr << "Error Reading Server Cert: " << server_entity.ssl_config().server_cert() << std::endl;
          std::cerr << "Exiting Program..." << std::endl;
          exit(1);
      }

      if (read_and_parse_file(server_key, temp_key) == -1) {
          std::cerr << "Error Reading Key Cert: " << server_entity.ssl_config().server_key() << std::endl;
          std::cerr << "Exiting Program..." << std::endl;
          exit(1);
      }

      std::cout << "SSL enabled" << std::endl;
      grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {server_key, server_cert};
      grpc::SslServerCredentialsOptions ssl_opts;
      ssl_opts.pem_root_certs = "";
      ssl_opts.pem_key_cert_pairs.push_back(pkcp);
      creds = grpc::SslServerCredentials(ssl_opts);

    } else {
      std::cout << "SSL disabled" << std::endl;
      creds = grpc::InsecureServerCredentials();
    }

    // Listens on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, creds);

    // Registers "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(service);

    // Override default grpc max received message size.
    builder.SetMaxReceiveMessageSize(INT_MAX);

    // Finally assemble the server.
    server_ = builder.BuildAndStart();
    std::cout << "Controller listening on " << server_address << std::endl;
  }

  void Stop() {
    if (server_ == nullptr) {
      return;
    }

    server_->Shutdown();
    //server_ = nullptr;
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
private:
  //Reads a file from disk that contains the key and certificate information
  //and returns the certificate and a referenced argument, or an error if file
  //is not being opened
  int read_and_parse_file(std::string &return_cert, std::string file_name){
    std::ifstream _file;
    _file.open(file_name);

    // Manage handling in case the certificates are not generated.
    std::stringstream buffer;
    if (_file.is_open()){
        buffer << _file.rdbuf();
        return_cert = buffer.str();
        return 1;
      }
    return -1;
  }
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
      const auto &params = controller_->GetParams();
      Start(params.server_entity(),this);
  }

  void WaitService() override { Wait(); }

  void StopService() override { Stop(); }

  Status GetCommunityModelEvaluationLineage(
      ServerContext *context,
      const GetCommunityModelEvaluationLineageRequest *request,
      GetCommunityModelEvaluationLineageResponse *response) override {

    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    const auto lineage = controller_->GetEvaluationLineage(
        request->num_backtracks());

    ModelEvaluations evaluations;
    for (const auto &evaluation : lineage) {
      *evaluations.add_evaluation() = evaluation;
    }

    *response->mutable_evaluations() = evaluations;
    return Status::OK;
  }

  Status GetLocalModelEvaluationLineage(
      ServerContext *context,
      const GetLocalModelEvaluationLineageRequest *request,
      GetLocalModelEvaluationLineageResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    for (const auto &learner_id : request->learner_ids()) {
      const auto lineage = controller_->GetEvaluationLineage(
          learner_id, request->num_backtracks());

      ModelEvaluations evaluations;
      for (const auto &evaluation : lineage) {
        *evaluations.add_evaluation() = evaluation;
      }

      (*response->mutable_learner_evaluations())[learner_id] = evaluations;
    }

    return Status::OK;
  }

  Status GetParticipatingLearners(
      ServerContext *context, const GetParticipatingLearnersRequest *request,
      GetParticipatingLearnersResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    // Creates LearnerEntity response collection.
    for (const auto &learner : controller_->GetLearners()) {
      std::cout << learner.DebugString() << std::endl;
      *response->add_server_entity() = learner.service_spec();
    }

    return Status::OK;
  }

  Status GetRuntimeMetadata(ServerContext *context,
                            const GetRuntimeMetadataRequest *request,
                            GetRuntimeMetadataResponse *response) override {
    *response->mutable_metadata() = controller_->RuntimeMetadata();
    return Status::OK;
  }

  Status JoinFederation(ServerContext *context,
                        const JoinFederationRequest *request,
                        JoinFederationResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    // Validates that the incoming request has the required fields populated.
    if (!request->has_server_entity() && !request->has_local_dataset_spec()) {
      response->mutable_ack()->set_status(false);
      return {StatusCode::INVALID_ARGUMENT,
              "Server entity and local dataset cannot be empty."};
    }

    const auto learner_or = controller_->AddLearner(
        request->server_entity(), request->local_dataset_spec());

    if (!learner_or.ok()) {
      response->mutable_ack()->set_status(false);
      // Returns the internal status error message as servicer's status message.
      return {StatusCode::ALREADY_EXISTS,
              std::string(learner_or.status().message())};
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
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    // Validates that the incoming request has the required fields populated.
    if (request->learner_id().empty() || request->auth_token().empty()) {
      response->mutable_ack()->set_status(false);
      return {StatusCode::INVALID_ARGUMENT,
              "Learner id and authentication token cannot be empty."};
    }

    const std::string &learner_id = request->learner_id();
    const std::string &auth_token = request->auth_token();

    const auto del_status = controller_->RemoveLearner(learner_id, auth_token);

    if (del_status.ok()) {
      response->mutable_ack()->set_status(true);
      return Status::OK;
    } else {
      response->mutable_ack()->set_status(false);
      return {StatusCode::CANCELLED, std::string(del_status.message())};
    }
  }

  Status MarkTaskCompleted(ServerContext *context,
                           const MarkTaskCompletedRequest *request,
                           MarkTaskCompletedResponse *response) override {
    const auto status = controller_->LearnerCompletedTask(
        request->learner_id(), request->auth_token(), request->task());
    if (!status.ok()) {
      switch (status.code()) {
      case absl::StatusCode::kInvalidArgument:
        response->mutable_ack()->set_status(false);
        return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
      case absl::StatusCode::kPermissionDenied:
        response->mutable_ack()->set_status(false);
        return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
      case absl::StatusCode::kNotFound:
        response->mutable_ack()->set_status(false);
        return {StatusCode::NOT_FOUND, std::string(status.message())};
      default:
        response->mutable_ack()->set_status(false);
        return {StatusCode::INTERNAL, std::string(status.message())};
      }
    }
    response->mutable_ack()->set_status(true);
    return Status::OK;
  }

  Status
  ReplaceCommunityModel(ServerContext *context,
                        const ReplaceCommunityModelRequest *request,
                        ReplaceCommunityModelResponse *response) override {
    auto status = controller_->ReplaceCommunityModel(request->model());
    if (status.ok()) {
      response->mutable_ack()->set_status(true);
      return Status::OK;
    }

    return {StatusCode::UNAUTHENTICATED, std::string(status.message())};
  }

  Status ShutDown(ServerContext *context, const ShutDownRequest *request,
                  ShutDownResponse *response) override {
    response->mutable_ack()->set_status(true);
    this->StopService();
    return Status::OK;
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
