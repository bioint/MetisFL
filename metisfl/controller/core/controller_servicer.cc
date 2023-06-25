
#include <csignal>
#include <future>
#include <memory>
#include <utility>

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "absl/memory/memory.h"
#include "metisfl/controller/core/controller_servicer.h"
#include "metisfl/controller/core/controller_utils.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/proto/controller.grpc.pb.h"
#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {
namespace {
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;

class ServicerBase {
 public:
  template<class Service>
  void Start(const ServerEntity &server_entity, Service *service) {
    const auto server_address = absl::StrCat(server_entity.hostname(), ":", server_entity.port());

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    std::shared_ptr<grpc::ServerCredentials> creds;

    if (server_entity.ssl_config().enable_ssl()) {

      std::string server_cert_loaded;
      std::string server_key_loaded;
      if (server_entity.ssl_config().has_ssl_config_files()) {
        auto cert_path = server_entity.ssl_config().ssl_config_files().public_certificate_file();
        auto key_path = server_entity.ssl_config().ssl_config_files().private_key_file();

        if (ReadParseFile(server_cert_loaded, cert_path) == -1) {
          // Logs and terminates the program if public certificate filepath is invalid.
          PLOG(FATAL) << "Error Reading Controller Certificate: " << cert_path;
        }

        if (ReadParseFile(server_key_loaded, key_path) == -1) {
          // Logs and terminates the program if private key filepath is invalid.
          PLOG(FATAL) << "Error Reading Controller Key: " << key_path;
        }
      } else if (server_entity.ssl_config().has_ssl_config_stream()) {
        server_cert_loaded = server_entity.ssl_config().ssl_config_stream().public_certificate_stream();
        server_key_loaded = server_entity.ssl_config().ssl_config_stream().private_key_stream();
      } else {
        PLOG(FATAL) << "Even though SSL was enabled the (private, public) key pair was not provided.";
      }

      PLOG(INFO) << "SSL enabled";

      grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp =
          {server_key_loaded, server_cert_loaded};
      grpc::SslServerCredentialsOptions ssl_opts;
      ssl_opts.pem_root_certs = "";
      ssl_opts.pem_key_cert_pairs.push_back(pkcp);
      creds = grpc::SslServerCredentials(ssl_opts);

    } else {
      PLOG(INFO) << "SSL disabled";
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
    PLOG(INFO) << "Controller listening on " << server_address;
  }

  void Stop() {
    if (server_ == nullptr) {
      return;
    }
    server_->Shutdown();
    PLOG(INFO) << "Controller shut down.";
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
      : pool_(1), controller_(controller) {
    GOOGLE_CHECK_NOTNULL(controller_);
  }

  ABSL_MUST_USE_RESULT
  const Controller *GetController() const override { return controller_; }

  void StartService() override {
    const auto &params = controller_->GetParams();
    Start(params.server_entity(), this);
    PLOG(INFO) << "Started Controller Servicer.";
  }

  void WaitService() override {
    Wait();
  }

  void StopService() override {
    pool_.push_task([this] { controller_->Shutdown(); });
    pool_.push_task([this] { this->Stop(); });
  }

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

    for (const auto &evaluation: lineage) {
      *response->add_community_evaluation() = evaluation;
    }

    return Status::OK;
  }

  Status GetLocalTaskLineage(
      ServerContext *context,
      const GetLocalTaskLineageRequest *request,
      GetLocalTaskLineageResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    for (const auto &learner_id: request->learner_ids()) {
      const auto lineage = controller_->GetLocalTaskLineage(
          learner_id, request->num_backtracks());

      LocalTasksMetadata learner_tasks_metadata;
      for (const auto &task_meta: lineage) {
        *learner_tasks_metadata.add_task_metadata() = task_meta;
      }
      (*response->mutable_learner_task())[learner_id] = learner_tasks_metadata;
    }

    return Status::OK;
  }

  Status GetRuntimeMetadataLineage(
      ServerContext *context,
      const GetRuntimeMetadataLineageRequest *request,
      GetRuntimeMetadataLineageResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    const auto lineage = controller_->GetRuntimeMetadataLineage(
        request->num_backtracks());

    for (const auto &metadata: lineage) {
      *response->add_metadata() = metadata;
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

    // Creates LearnerDescriptor response collection that only
    // contains the learner's id and dataset specifications.
    for (const auto &learner: controller_->GetLearners()) {
      LearnerDescriptor learner_descriptor;
      *learner_descriptor.mutable_id() = learner.id();
      *learner_descriptor.mutable_dataset_spec() = learner.dataset_spec();
      *response->add_learner() = learner_descriptor;
    }

    return Status::OK;
  }

  Status GetServicesHealthStatus(
      ServerContext *context, const GetServicesHealthStatusRequest *request,
      GetServicesHealthStatusResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }
    // TODO(stripeli): We need to capture the heartbeat of all the
    //  underlying controller services. Controller being null is one thing
    //  but if any of its services is not available (e.g., model store, aggregation)
    //  then we need to return a corresponding health check.
    if (controller_ != nullptr) {
      (*response->mutable_services_status())["controller"] = true;
    } else {
      (*response->mutable_services_status())["controller"] = false;
    }
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
      switch (learner_or.status().code()) {
        case absl::StatusCode::kAlreadyExists:
          return {StatusCode::ALREADY_EXISTS,
                  std::string(learner_or.status().message())};
        default:
          return {StatusCode::INVALID_ARGUMENT,
                  std::string(learner_or.status().message())};
      }
    } else {
      response->mutable_ack()->set_status(true);

      const auto &learner_state = learner_or.value();
      response->set_learner_id(learner_state.id());
      response->set_auth_token(learner_state.auth_token());
    }

    PLOG(INFO) << "Learner " << learner_or.value().id() << " joined Federation.";
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
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }

    PLOG(INFO) << "Received Completed Task By " << request->learner_id();
    const auto status = controller_->LearnerCompletedTask(
        request->learner_id(), request->auth_token(), request->task());
    if (!status.ok()) {
      switch (status.code()) {
        case absl::StatusCode::kInvalidArgument:response->mutable_ack()->set_status(false);
          return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
        case absl::StatusCode::kPermissionDenied:response->mutable_ack()->set_status(false);
          return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
        case absl::StatusCode::kNotFound:response->mutable_ack()->set_status(false);
          return {StatusCode::NOT_FOUND, std::string(status.message())};
        default:response->mutable_ack()->set_status(false);
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
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }
    auto status = controller_->ReplaceCommunityModel(request->model());
    if (status.ok()) {
      response->mutable_ack()->set_status(true);
      PLOG(INFO) << "Replaced Community Model.";
      return Status::OK;
    }

    PLOG(ERROR) << "Couldn't Replace Community Model.";
    return {StatusCode::UNAUTHENTICATED, std::string(status.message())};
  }

  Status ShutDown(ServerContext *context, const ShutDownRequest *request,
                  ShutDownResponse *response) override {
    // Captures unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return {StatusCode::INVALID_ARGUMENT,
              "Request and response cannot be empty."};
    }
    response->mutable_ack()->set_status(true);
    this->StopService();
    return Status::OK;
  }

 private:
  // Thread pool for async tasks.
  BS::thread_pool pool_;
  Controller *controller_;
};
} // namespace

std::unique_ptr<ControllerServicer>
ControllerServicer::New(Controller *controller) {
  return absl::make_unique<ControllerServicerImpl>(controller);
}

} // namespace metisfl::controller
