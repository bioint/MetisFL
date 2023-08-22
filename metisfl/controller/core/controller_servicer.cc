
#include "metisfl/controller/core/controller_servicer.h"

namespace metisfl::controller {
namespace {
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;

class ControllerServicerImpl : public ControllerServicer {
 public:
  explicit ControllerServicerImpl(ServerParams &server_params,
                                  Controller *controller)
      : pool_(1), controller_(controller), server_params_(params) {
    GOOGLE_CHECK_NOTNULL(controller_);
  }

  const Controller *GetController() const override { return controller_; }

  void StartService() override {
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    std::shared_ptr<grpc::ServerCredentials> creds;

    const auto &root = server_params_.root_certificate;
    const auto &public = server_params_.public_certificate;
    const auto &private = server_params_.private_key;

    auto ssl_enable = !root.empty() && !public.empty() && !private.empty();

    if (ssl_enable) {
      std::string root_certificate;
      std::string server_certifcate;
      std::string private_key;

      ReadParseFile(root_certificate, root);
      ReadParseFile(server_certifcate, public);
      ReadParseFile(private_key, private);

      grpc::SslServerCredentialsOptions ssl_opts;
      ssl_opts.pem_root_certs = root_certificate;
      grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp = {
          private_key, server_certifcate};
      ssl_opts.pem_key_cert_pairs.push_back(pkcp);
      creds = grpc::SslServerCredentials(ssl_opts);
    } else
      creds = grpc::InsecureServerCredentials();

    const auto server_address =
        absl::StrCat(server_params_.hostname, ":", server_params_.port);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, creds);
    builder.RegisterService(this);
    builder.SetMaxReceiveMessageSize(INT_MAX);
    server_ = builder.BuildAndStart();

    if (ssl_enable)
      PLOG(INFO) << "Controller listening on " << server_address
                 << " with SSL enabled.";
    else
      PLOG(INFO) << "Controller listening on " << server_address << ".";
  }

  void WaitService() override {
    if (server_ == nullptr) return;

    server_->Wait();
  }

  void StopService() override {
    pool_.push_task([this] { controller_->Shutdown(); });
    pool_.push_task([this] { this->Stop(); });
  }

  void Stop() {
    if (server_ == nullptr) return;

    server_->Shutdown();
    PLOG(INFO) << "Controller shut down.";
  }

  bool ShutdownRequestReceived() override { return shutdown_; }

  Status GetHealthStatus(ServerContext *context, const Empty *request,
                         Ack *ack) override {
    bool status = controller_ != nullptr;
    ack->set_status(status);
    return Status::OK;
  }

  Status JoinFederation(ServerContext *context,
                        const LearnerDescriptor *learner,
                        LearnerId *learnerId) override {
    if (learner->hostname().empty() || learner->port() <= 0 ||
        learner->num_training_examples() <= 0) {
      return {StatusCode::INVALID_ARGUMENT,
              "Must provide a valid hostname, port and number of training "
              "examples."};
    }

    const auto &learner_id = controller_->AddLearner(*learner);

    if (!learner_id.ok()) {
      switch (learner_id.status().code()) {
        case absl::StatusCode::kAlreadyExists:
          return {StatusCode::ALREADY_EXISTS,
                  std::string(learner_id.status().message())};
        default:
          return {StatusCode::INVALID_ARGUMENT,
                  std::string(learner_id.status().message())};
      }
    } else
      learnerId->set_id(learner_id.value());

    PLOG(INFO) << "Learner " << learner_id.value() << " joined Federation.";
    return Status::OK;
  }

  Status StartTraining(ServerContext *context, const Empty *request,
                       Ack *ack) override {
    const auto status = controller_->StartTraining();
    ack->set_status(true);
    return Status::OK;
  }

  Status LeaveFederation(ServerContext *context, const LearnerId *learnerId,
                         Ack *ack) override {
    if (learnerId->id().empty()) {
      ack->set_status(false);
      return {StatusCode::INVALID_ARGUMENT, "Learner id  cannot be empty."};
    }

    const auto del_status = controller_->RemoveLearner(learnerId->id());

    if (del_status.ok()) {
      ack->set_status(true);
      return Status::OK;
    } else {
      ack->set_status(false);
      return {StatusCode::CANCELLED, std::string(del_status.message())};
    }
  }

  Status TrainDone(ServerContext *context, const TrainDoneRequest *request,
                   Ack *ack) override {
    PLOG(INFO) << "Received Completed Task By " << request->learner_id();
    const auto status = controller_->TrainDone(*request);
    if (!status.ok()) {
      switch (status.code()) {
        case absl::StatusCode::kInvalidArgument:
          ack->set_status(false);
          return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
        case absl::StatusCode::kPermissionDenied:
          ack->set_status(false);
          return {StatusCode::PERMISSION_DENIED, std::string(status.message())};
        case absl::StatusCode::kNotFound:
          ack->set_status(false);
          return {StatusCode::NOT_FOUND, std::string(status.message())};
        default:
          ack->set_status(false);
          return {StatusCode::INTERNAL, std::string(status.message())};
      }
    }
    ack->set_status(true);
    return Status::OK;
  }

  Status SetInitialModel(ServerContext *context, const Model *model,
                         Ack *ack) override {
    auto status = controller_->SetInitialModel(*model);
    if (status.ok()) {
      PLOG(INFO) << "Replaced Community Model.";
      ack->set_status(true);
      return Status::OK;
    }

    PLOG(ERROR) << "Couldn't Replace Community Model.";
    return {StatusCode::UNAUTHENTICATED, std::string(status.message())};
  }

  Status ShutDown(ServerContext *context, const Empty *request,
                  Ack *ack) override {
    shutdown_ = true;
    ack->set_status(true);
    this->StopService();
    return Status::OK;
  }
};
}  // namespace

std::unique_ptr<ControllerServicer> ControllerServicer::New(
    ServerParams &params, Controller *controller) {
  return absl::make_unique<ControllerServicerImpl>(controller);
}

}  // namespace metisfl::controller
