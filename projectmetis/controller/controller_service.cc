//
// Created by Chrysovalantis Anastasiou on 3/10/21.
//

#include "projectmetis/controller/controller_service.h"

#include <memory>

#include "projectmetis/proto/controller.grpc.pb.h"
#include "projectmetis/proto/shared.grpc.pb.h"

namespace projectmetis::controller {
using ::grpc::Status;
using ::grpc::ServerContext;

class ControllerServiceImpl final : public Controller::Service {

public:
  Status JoinFederation(ServerContext* context,
                        const JoinFederationRequest* request,
                        JoinFederationResponse* response) override {

    // TODO(canastas) I removed the `context == nullptr` check because all
    //  service tests have the ServerContext equal to `nullptr`.
    // Capture unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return Status::CANCELLED;
    }

    // Validate that the incoming request has the required fields populated.
    if (!request->has_learner_entity() && !request->has_local_dataset_spec()) {
      response->mutable_ack()->set_status(false);
      return Status::OK;
    }

    // Validate non-empty hostname and non-negative port.
    if (request->learner_entity().hostname().empty() ||
        request->learner_entity().port() < 0) {
      response->mutable_ack()->set_status(false);
      return Status::OK;
    }

    // Validate number of train, validation and test examples. Train examples
    // must always be positive, while validation and test can be non-negative.
    if (request->local_dataset_spec().num_training_examples() <= 0 ||
        request->local_dataset_spec().num_validation_examples() < 0 ||
        request->local_dataset_spec().num_test_examples() < 0) {
      response->mutable_ack()->set_status(false);
      return Status::OK;
    }

    const std::string &hostname = request->learner_entity().hostname();
    const int32_t port = request->learner_entity().port();

    // TODO(dstripelis) Condition to ping the hostname + port.

    // Generate learner id.
    const std::string learner_id = CreateLearnerId(hostname, port);

    // Register learner to state map.
    if (!learner_state_map_.count(learner_id)) {
      // TODO(canastas) We need a better authorization token generator.
      const std::string auth_token = std::to_string(GetNumLearners() + 1);

      // Initialize learner state with hostname, port and authorization token.
      // Since learner just joined the federation, the federated model is empty.
      LearnerState learner_state;
      *learner_state.mutable_learner_entity() = request->learner_entity();
      learner_state.set_auth_token(auth_token);
      *learner_state.mutable_local_dataset_spec() =
          request->local_dataset_spec();
      learner_state_map_[learner_id] = learner_state;

      // Construct rpc response.
      response->set_learner_id(learner_id);
      response->set_auth_token(auth_token);
      response->mutable_ack()->set_status(true);
    } else {
      // Learner has already registered with the controller.
      response->mutable_ack()->set_status(false);
    }
    return Status::OK;
  }

  Status LeaveFederation(ServerContext *context,
                         const LeaveFederationRequest *request,
                         LeaveFederationResponse *response) override {

    // TODO(canastas) I removed the `context == nullptr` check because all
    //  service tests have the ServerContext equal to `nullptr`.
    // Capture unexpected behavior.
    if (request == nullptr || response == nullptr) {
      return Status::CANCELLED;
    }

    // Validate that the incoming request has the required fields populated.
    if (request->learner_id().empty() || request->auth_token().empty()) {
      response->mutable_ack()->set_status(false);
      return Status::OK;
    }

    const std::string &learner_id = request->learner_id();
    const std::string &auth_token = request->auth_token();

    auto it = learner_state_map_.find(learner_id);
    // Check requesting learner existence inside the state map.
    if (it != learner_state_map_.end()) {
      if (it->second.auth_token() == auth_token) {
        learner_state_map_.erase(it);
        // Learner exists, success.
        response->mutable_ack()->set_status(true);
        return Status::OK;
      } else {
        // Learner exists but it has wrong credentials (token), failure.
        response->mutable_ack()->set_status(false);
        return Status::OK;
      }
    } else {
      // Learner does not exist so there is not much we can do.
      response->mutable_ack()->set_status(false);
      return Status::OK;
    }
  }

private:
  // A lookup map for learners execution state, based on the definition of the
  // LearnerState proto message.
  std::unordered_map<std::string, LearnerState> learner_state_map_;

  int GetNumLearners() const {
    return learner_state_map_.size();
  }

  static std::string CreateLearnerId(const std::string &hostname,
                                     const int &port) {
    /**
     * Generate the unique identifier of each learner. Learners can originate
     * from the same server (hostname) but listening to different ports.
     * To avoid any conflicts, we must create a unique id/key that is a
     * combination of the learner's hostname and port.
     */
    return hostname + ":" + std::to_string(port);
  }

};

std::unique_ptr<Controller::Service> New() {
  return std::make_unique<ControllerServiceImpl>();
}

}  // namespace projectmetis::controller
