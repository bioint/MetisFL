
#include "metisfl/controller/core/controller_servicer.h"

namespace metisfl::controller {

void ControllerServicer::StartService() {
  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  std::shared_ptr<grpc::ServerCredentials> creds;

  const auto &root_cert = server_params_.root_certificate;
  const auto &public_cert = server_params_.public_certificate;
  const auto &private_cert = server_params_.private_key;

  auto ssl_enable =
      !root_cert.empty() && !public_cert.empty() && !private_cert.empty();

  if (ssl_enable) {
    std::string root_certificate;
    std::string server_certifcate;
    std::string private_key;

    ReadParseFile(root_certificate, root_cert);
    ReadParseFile(server_certifcate, public_cert);
    ReadParseFile(private_key, private_cert);

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
    LOG(INFO) << "Controller listening on " << server_address
              << " with SSL enabled.";
  else
    LOG(INFO) << "Controller listening on " << server_address << ".";
}

void ControllerServicer::WaitService() {
  if (server_ == nullptr) return;

  server_->Wait();
}

void ControllerServicer::StopService() {
  pool_.push_task([this] { controller_->Shutdown(); });
  pool_.push_task([this] { this->ShutdownServer(); });
}

void ControllerServicer::ShutdownServer() {
  if (server_ == nullptr) return;

  server_->Shutdown();
  LOG(INFO) << "Controller shut down.";
}

bool ControllerServicer::ShutdownRequestReceived() { return shutdown_; }

Status ControllerServicer::GetHealthStatus(ServerContext *context,
                                           const Empty *request, Ack *ack) {
  bool status = controller_ != nullptr;
  ack->set_status(status);
  return Status::OK;
}

Status ControllerServicer::JoinFederation(ServerContext *context,
                                          const Learner *learner,
                                          LearnerId *learnerId) {
  if (learner->hostname().empty() || learner->port() <= 0) {
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

  LOG(INFO) << "Learner " << learner_id.value() << " joined Federation.";
  return Status::OK;
}

Status ControllerServicer::SetInitialModel(ServerContext *context,
                                           const Model *model, Ack *ack) {
  auto status = controller_->SetInitialModel(*model);
  if (status.ok()) {
    LOG(INFO) << "Received Initial Model.";
    ack->set_status(true);
    return Status::OK;
  }
  // TODO: better error handling
  LOG(ERROR) << "Couldn't Replace Initial Model.";
  return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
}

Status ControllerServicer::StartTraining(ServerContext *context,
                                         const Empty *request, Ack *ack) {
  const auto status = controller_->StartTraining();
  if (status.ok()) {
    LOG(INFO) << "Started Training.";
    ack->set_status(true);
    return Status::OK;
  } else {
    ack->set_status(false);
    return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
  }
}

Status ControllerServicer::LeaveFederation(ServerContext *context,
                                           const LearnerId *learnerId,
                                           Ack *ack) {
  if (learnerId->id().empty()) {
    ack->set_status(false);
    return {StatusCode::INVALID_ARGUMENT, "Learner id  cannot be empty."};
  }

  const auto del_status = controller_->RemoveLearner(learnerId->id());

  if (del_status.ok()) {
    LOG(INFO) << "Learner " << learnerId->id() << " left Federation.";
    ack->set_status(true);
    return Status::OK;
  } else {
    ack->set_status(false);
    return {StatusCode::CANCELLED, std::string(del_status.message())};
  }
}

Status ControllerServicer::TrainDone(ServerContext *context,
                                     const TrainDoneRequest *request,
                                     Ack *ack) {
  LOG(INFO) << "Received Completed Task from Learner "
            << controller_->GetLearnerId(request->task().id());

  const auto status = controller_->TrainDone(*request);

  if (!status.ok()) {
    switch (status.code()) {
      case absl::StatusCode::kInvalidArgument:
        ack->set_status(false);
        return {StatusCode::INVALID_ARGUMENT, std::string(status.message())};
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

Status ControllerServicer::GetLogs(ServerContext *context, const Empty *request,
                                   Logs *logs) {
  auto tasks_map = controller_->GetTaskMap();
  auto train_results = controller_->GetTrainResults();
  auto evaluation_results = controller_->GetEvaluationResults();
  auto model_metadata = controller_->GetModelMetadata();

  auto global_iteration = controller_->GetGlobalIteration();
  if (global_iteration > 0) logs->set_global_iteration(global_iteration);

  for (auto &task : tasks_map) {
    auto *task_proto = logs->add_tasks();
    *task_proto = task.second;
  }

  *logs->mutable_train_results() =
      google::protobuf::Map<std::string, TrainResults>(train_results.begin(),
                                                       train_results.end());

  *logs->mutable_evaluation_results() =
      google::protobuf::Map<std::string, EvaluationResults>(
          evaluation_results.begin(), evaluation_results.end());

  *logs->mutable_model_metadata() =
      google::protobuf::Map<std::string, ModelMetadata>(model_metadata.begin(),
                                                        model_metadata.end());
  return Status::OK;
}

Status ControllerServicer::ShutDown(ServerContext *context,
                                    const Empty *request, Ack *ack) {
  shutdown_ = true;
  ack->set_status(true);
  this->StopService();
  return Status::OK;
}
}  // namespace metisfl::controller
