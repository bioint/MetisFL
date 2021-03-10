//
// Created by Chrysovalantis Anastasiou on 3/10/21.
//

#include "projectmetis/controller/controller_service.h"

#include <memory>

#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {
using ::grpc::Status;
using ::grpc::ServerContext;

class ControllerServiceImpl final : public Controller::Service {
  Status JoinFederation(ServerContext* context,
                        const JoinFederationRequest* request,
                        JoinFederationResponse* response) override {
    response->set_learner_id("learner1");
    response->set_auth_token("qwerty");
    response->mutable_ack()->set_status(true);
    return Status::OK;
  }
};

std::unique_ptr<Controller::Service> New() {
  return std::make_unique<ControllerServiceImpl>();
}

}  // namespace projectmetis::controller