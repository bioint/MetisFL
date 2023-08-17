
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

namespace metisfl::controller
{
  namespace
  {
    using ::grpc::Server;
    using ::grpc::ServerBuilder;
    using ::grpc::ServerContext;
    using ::grpc::Status;
    using ::grpc::StatusCode;

    class ServicerBase
    {
    public:
      template <class Service>
      void Start(const std::string &hostname,
                 const int port,
                 const std::string &public_certifcate_file,
                 const std::string &private_key_file,
                 Service *service)
      {
        const auto server_address = absl::StrCat(hostname, ":", port); // TODO: make sure int is ok

        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();

        ServerBuilder builder;
        std::shared_ptr<grpc::ServerCredentials> creds;

        auto ssl_enable = !public_certifcate_file.empty() && !private_key_file.empty();

        if (ssl_enable)
        {
          std::string server_cert_loaded;
          std::string server_key_loaded;

          if (ReadParseFile(server_cert_loaded, public_certifcate_file) == -1)
            PLOG(FATAL) << "Error reading controller certificate: " << cert_path;

          if (ReadParseFile(server_key_loaded, private_key_file) == -1)
            PLOG(FATAL) << "Error reading controller key: " << key_path;

          PLOG(INFO) << "SSL enabled.";

          grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp =
              {server_key_loaded, server_cert_loaded};
          grpc::SslServerCredentialsOptions ssl_opts;
          ssl_opts.pem_root_certs = ""; // FIXME: what about root certs?
          ssl_opts.pem_key_cert_pairs.push_back(pkcp);
          creds = grpc::SslServerCredentials(ssl_opts);
        }
        else
        {
          PLOG(INFO) << "SSL disabled";
          creds = grpc::InsecureServerCredentials();
        }

        builder.AddListeningPort(server_address, creds);
        builder.RegisterService(service);
        builder.SetMaxReceiveMessageSize(INT_MAX);
        server_ = builder.BuildAndStart();

        PLOG(INFO) << "Controller listening on " << server_address;
      }

      void Stop()
      {
        if (server_ == nullptr)
          return;

        server_->Shutdown();
        PLOG(INFO) << "Controller shut down.";
      }

      void Wait()
      {
        if (server_ == nullptr)
          return;

        server_->Wait();
      }

    protected:
      std::unique_ptr<Server> server_;
    };

    class ControllerServicerImpl : public ControllerServicer, private ServicerBase
    {
    public:
      explicit ControllerServicerImpl(Controller *controller)
          : pool_(1), controller_(controller)
      {
        GOOGLE_CHECK_NOTNULL(controller_);
      }

      ABSL_MUST_USE_RESULT
      const Controller *GetController() const override { return controller_; }

      void StartService() override
      {
        const auto &params = controller_->GetParams();
        Start(params.server_entity(), this);
        PLOG(INFO) << "Started Controller Servicer.";
      }

      void WaitService() override
      {
        Wait();
      }

      void StopService() override
      {
        pool_.push_task([this]
                        { controller_->Shutdown(); });
        pool_.push_task([this]
                        { this->Stop(); });
      }

      bool ShutdownRequestReceived() override
      {
        return shutdown_;
      }

      Status GetStatistics(
          ServerContext *context,
          const GetStatisticsRequest *request,
          GetStatisticsResponse *response) override
      {

        // Learner ids
        for (const auto &learner : controller_->GetLearners())
          *response->add_learner_id() = learner.id();

        // Evaluation lineage
        const auto eval_lineage = controller_->GetEvaluationLineage(
            request->community_evaluation_backtracks());
        for (const auto &evaluation : eval_lineage)
          *response->add_community_evaluation() = evaluation;

        // Task lineage
        for (const auto &learner_id : request->learner_ids())
        {
          const auto lineage = controller_->GetLocalTaskLineage(
              learner_id, request->local_task_backtracks());

          LocalTasksMetadata learner_tasks_metadata;
          for (const auto &task_meta : lineage)
            *learner_tasks_metadata.add_task_metadata() = task_meta;

          (*response->mutable_learner_task())[learner_id] = learner_tasks_metadata;
        }

        // Runtime metadata
        const auto task_lineage = controller_->GetRuntimeMetadataLineage(
            request->metadata_backtracks());
        for (const auto &metadata : task_lineage)
          *response->add_metadata() = metadata;

        return Status::OK;
      }

      Status GetHealthStatus(
          ServerContext *context,
          const Empty *request,
          Ack *response) override
      {
        response->set_status(true);
        return Status::OK;
      }

      Status JoinFederation(ServerContext *context,
                            const JoinFederationRequest *request,
                            JoinFederationResponse *response) override
      {
        if (request->hostname().empty() || request->port() <= 0)
        {
          response->mutable_ack()->set_status(false);
          return {StatusCode::INVALID_ARGUMENT,
                  "Server entity or local dataset cannot be empty."};
        }

        const auto learner_or = controller_->AddLearner(
            request->hostname(), request->port(), request->public_certificate_bytes(), request->num_traning_examples());

        if (!learner_or.ok())
        {
          response->mutable_ack()->set_status(false);
          switch (learner_or.status().code())
          {
          case absl::StatusCode::kAlreadyExists:
            return {StatusCode::ALREADY_EXISTS,
                    std::string(learner_or.status().message())};
          default:
            return {StatusCode::INVALID_ARGUMENT,
                    std::string(learner_or.status().message())};
          }
        }
        else
        {
          const auto &learner_state = learner_or.value();
          response->set_learner_id(learner_state.id());
          response->set_auth_token(learner_state.auth_token());
        }

        PLOG(INFO) << "Learner " << learner_or.value().id() << " joined Federation.";
        return Status::OK;
      }

      Status LeaveFederation(ServerContext *context,
                             const LeaveFederationRequest *request,
                             Ack *response) override
      {

        if (request->learner_id().empty() || request->auth_token().empty())
        {
          response->mutable_ack()->set_status(false);
          return {StatusCode::INVALID_ARGUMENT,
                  "Learner id and authentication token cannot be empty."};
        }

        const std::string &learner_id = request->learner_id();
        const std::string &auth_token = request->auth_token();

        const auto del_status = controller_->RemoveLearner(learner_id, auth_token);

        if (del_status.ok())
        {
          response->set_status(true);
          return Status::OK;
        }
        else
        {
          response->set_status(false);
          return {StatusCode::CANCELLED, std::string(del_status.message())};
        }
      }

      Status TrainDone(ServerContext *context,
                       const TrainDoneRequest *request,
                       Ack *response) override
      {
        PLOG(INFO) << "Received Completed Task By " << request->learner_id();
        const auto status = controller_->LearnerCompletedTask(
            request->learner_id(), request->auth_token(), request->task());
        if (!status.ok())
        {
          switch (status.code())
          {
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
        response->set_status(true);
        return Status::OK;
      }

      Status
      SetInitialModel(ServerContext *context,
                      const Model *model,
                      Ack *response) override
      {
        auto status = controller_->SetInitialModel(model);
        if (status.ok())
        {
          response->set_status(true);
          PLOG(INFO) << "Replaced Community Model.";
          return Status::OK;
        }

        PLOG(ERROR) << "Couldn't Replace Community Model.";
        return {StatusCode::UNAUTHENTICATED, std::string(status.message())};
      }

      Status ShutDown(ServerContext *context,
                      const Empty *request,
                      Ack *response) override
      {
        shutdown_ = true;
        response->mutable_ack()->set_status(true);
        this->StopService();
        return Status::OK;
      }

    private:
      BS::thread_pool pool_;
      Controller *controller_;
      bool shutdown_ = false;
    };
  } // namespace

  std::unique_ptr<ControllerServicer>
  ControllerServicer::New(Controller *controller)
  {
    return absl::make_unique<ControllerServicerImpl>(controller);
  }

} // namespace metisfl::controller
