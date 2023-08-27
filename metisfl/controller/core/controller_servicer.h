
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <utility>

#include "absl/memory/memory.h"
#include "metisfl/controller/common/bs_thread_pool.h"
#include "metisfl/controller/core/controller.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/proto/controller.grpc.pb.h"

using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;

namespace metisfl::controller {
class ControllerServicer final : public ControllerService::Service {
  std::unique_ptr<Server> server_;
  ServerParams server_params_;
  BS::thread_pool pool_;
  Controller* controller_;
  bool shutdown_ = false;

 public:
  ControllerServicer(const ServerParams& server_params, Controller* controller)
      : server_params_(server_params), pool_(1), controller_(controller){};

  void StartService();

  void WaitService();

  void StopService();

  void ShutdownServer();

  bool ShutdownRequestReceived();

  Status GetHealthStatus(ServerContext* context, const metisfl::Empty* request,
                         metisfl::Ack* response) override;
  Status SetInitialModel(ServerContext* context, const metisfl::Model* request,
                         metisfl::Ack* response) override;
  Status JoinFederation(ServerContext* context, const metisfl::Learner* request,
                        metisfl::LearnerId* response) override;
  Status LeaveFederation(ServerContext* context,
                         const metisfl::LearnerId* request,
                         metisfl::Ack* response) override;
  Status StartTraining(ServerContext* context, const metisfl::Empty* request,
                       metisfl::Ack* response) override;
  Status TrainDone(ServerContext* context,
                   const metisfl::TrainDoneRequest* request,
                   metisfl::Ack* response) override;
  Status GetLogs(ServerContext* context, const metisfl::Empty* request,
                 metisfl::Logs* response) override;
  Status ShutDown(ServerContext* context, const metisfl::Empty* request,
                  metisfl::Ack* response) override;
};
}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_SERVICER_H_
