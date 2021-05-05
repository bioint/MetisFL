#include "projectmetis/controller/controller_service.h"

#include <gtest/gtest.h>

namespace projectmetis {
namespace {

TEST(ControllerService, JoinFederationEmptyRequest) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, JoinFederationHasHostButEmptyPort) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, JoinFederationHasPortButEmptyHost) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_port(1991);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, JoinFederationLearnerServiceNotReachable) /* NOLINT */ {
  auto service = controller::New();

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(1991);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_FALSE(status.ok());
}

TEST(ControllerService, JoinFederationSuccess) /* NOLINT */ {
  auto service = controller::New();

  // TODO(canastas): start a temporary service here

  JoinFederationRequest req;
  req.mutable_learner_entity()->set_hostname("localhost");
  req.mutable_learner_entity()->set_port(1991);
  JoinFederationResponse res;

  auto status = service->JoinFederation(nullptr, &req, &res);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res.ack().status());
  EXPECT_FALSE(res.auth_token().empty());
}

}  // namespace
}  // namespace projectmetis