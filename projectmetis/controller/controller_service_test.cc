#include "projectmetis/controller/controller_service.h"

#include <gtest/gtest.h>

namespace projectmetis {
namespace {

TEST(ControllerService, JoinFederation) {
  auto service = controller::New();

  JoinFederationRequest req;
  JoinFederationResponse res;

  service->JoinFederation(nullptr, &req, &res);

  EXPECT_TRUE(res.ack().status());
}

}  // namespace
}  // namespace projectmetis