#include "projectmetis/controller/controller_service.h"

#include <gtest/gtest.h>

namespace projectmetis {
namespace {

TEST(ControllerService, GetParticipatingLearnersEmptyFederation) /* NOLINT */ {
  auto service = controller::New();

  GetParticipatingLearnersRequest req;
  GetParticipatingLearnersResponse res;

  auto status = service->GetParticipatingLearners(nullptr, &req, &res);

  EXPECT_TRUE(status.ok());
}

TEST(ControllerService, GetParticipatingLearnersLoadedFederation) /* NOLINT */ {
  auto service = controller::New();

  std::string hostname = "localhost1";
  int32_t port = 1991;

  // TODO(canastas) We need the learner micro service spawned in this test too.

  // Register first learner with the controller.
  JoinFederationRequest join_req;
  join_req.mutable_learner_entity()->set_hostname(hostname);
  join_req.mutable_learner_entity()->set_port(port);
  join_req.mutable_local_dataset_spec()->set_num_training_examples(100);
  JoinFederationResponse join_res;

  auto join_status = service->JoinFederation(nullptr, &join_req, &join_res);

  GetParticipatingLearnersRequest get_req;
  GetParticipatingLearnersResponse get_res;

  auto get_status =
      service->GetParticipatingLearners(nullptr, &get_req, &get_res);

  // We only registered a single learner hence the check of size equal to 1.
  EXPECT_EQ(get_res.learner_entity().size(), 1);
  EXPECT_EQ(get_res.mutable_learner_entity(0)->hostname(), hostname);
  EXPECT_EQ(get_res.mutable_learner_entity(0)->port(), port);
  EXPECT_TRUE(get_status.ok());
}

TEST(ControllerService, JoinFederationEmptyRequest) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, JoinFederationHasHostButNegativePort) /* NOLINT */ {
  /**
   * In proto3 all messages have a default value. For the case of an int_32 type
   * the default value is 0. Since 0 is a valid port even though unconventional
   * for a service to run, we need to validate that no negative value is passed.
   */
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(-1234);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, JoinFederationHasPortButEmptyHost) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_port(1991);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, JoinFederationHasNegativeDatasetSpec) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(1991);
  req.mutable_local_dataset_spec()->set_num_training_examples(-1);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, JoinFederationLearnerServiceNotReachable) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(1991);
  req.mutable_local_dataset_spec()->set_num_training_examples(100);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, JoinFederationSuccess) /* NOLINT */ {
  auto service = controller::New();

  // TODO(canastas): start a temporary service here

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(1991);
  req.mutable_local_dataset_spec()->set_num_training_examples(100);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res.ack().status());
  EXPECT_FALSE(res.learner_id().empty());
  EXPECT_FALSE(res.auth_token().empty());
}

TEST(ControllerService, LeaveFederationEmptyRequest) /* NOLINT */ {
  auto service = controller::New();

  LeaveFederationRequest req;
  LeaveFederationResponse res;

  auto status = service->LeaveFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, LeaveFederationEmptyLearnerID) /* NOLINT */ {
  auto service = controller::New();

  LeaveFederationRequest req;
  req.set_auth_token("1234");
  LeaveFederationResponse res;

  auto status = service->LeaveFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, LeaveFederationEmptyAuthToken) /* NOLINT */ {
  auto service = controller::New();

  LeaveFederationRequest req;
  req.set_learner_id("localhost");
  LeaveFederationResponse res;

  auto status = service->LeaveFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, LeaveFederationNonRegisteredLearner) /* NOLINT */ {
  auto service = controller::New();

  LeaveFederationRequest req;
  req.set_learner_id("localhost");
  req.set_auth_token("1234");
  LeaveFederationResponse res;

  auto status = service->LeaveFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

TEST(ControllerService, LeaveFederationSucess) /* NOLINT */ {
  /**
   * First we need to register the new learner with the controller/federation,
   * hence the first JoinFederationRequest and then we need to ask from the
   * controller to exclude the learner from the federation.
   */
  auto service = controller::New();

  // TODO(canastas): start a temporary service here

  JoinFederationRequest join_req;
  join_req.mutable_learner_entity()->set_hostname("localhost");
  join_req.mutable_learner_entity()->set_port(1991);
  join_req.mutable_local_dataset_spec()->set_num_training_examples(100);
  JoinFederationResponse join_res;

  auto join_status = service->JoinFederation(nullptr, &join_req, &join_res);

  LeaveFederationRequest leave_req;
  leave_req.set_learner_id(join_res.learner_id());
  leave_req.set_auth_token(join_res.auth_token());
  LeaveFederationResponse leave_res;
  auto leave_status = service->LeaveFederation(nullptr, &leave_req, &leave_res);

  EXPECT_TRUE(leave_status.ok());
  EXPECT_TRUE(leave_res.ack().status());
}

}  // namespace
}  // namespace projectmetis
