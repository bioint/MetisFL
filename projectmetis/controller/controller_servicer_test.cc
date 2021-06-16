// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "projectmetis/controller/controller_servicer.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "projectmetis/core/macros.h"
#include "projectmetis/core/matchers/proto_matchers.h"

namespace projectmetis::controller {
namespace {
using ::grpc::ServerContext;
using ::proto::ParseTextOrDie;
using ::testing::Exactly;
using ::testing::Return;
using ::testing::proto::EqualsProto;
using ::testing::_;

const char kLearnerState[] = R"pb(
  learner_id: "localhost:1991"
  server_entity {
    hostname: "localhost"
    port: 1991
  }
  auth_token: "token"
  local_dataset_spec {
    num_training_examples: 1
    num_validation_examples: 1
    num_test_examples: 1
  }
)pb";

class MockController : public Controller {
 public:
  MOCK_METHOD(ControllerParams &, GetParams, (), (const));
  MOCK_METHOD(std::vector<LearnerState>, GetLearners, (), (const));
  MOCK_METHOD(absl::StatusOr<LearnerState>, AddLearner,
              (const ServerEntity &, const LocalDatasetSpec &));
  MOCK_METHOD(absl::Status, RemoveLearner,
              (const std::string &, const std::string &));
};

class ControllerServicerImplTest : public ::testing::Test {
 protected:
  ServerContext ctx_;

  MockController controller_;
  std::unique_ptr<ControllerServicer> service_ =
      ControllerServicer::New(&controller_);
};

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_EmptyRequest) {
  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;
  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_EmptyVector) {
  EXPECT_CALL(controller_, GetLearners())
      .Times(Exactly(1))
      .WillOnce(Return(std::vector<LearnerState>()));

  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;
  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res_.server_entity().empty());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, GetParticipatingLearners_NotEmptyVector) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_, GetLearners())
      .Times(Exactly(1))
      .WillOnce(Return(std::vector({learner_state})));

  GetParticipatingLearnersRequest req_;
  GetParticipatingLearnersResponse res_;
  auto status = service_->GetParticipatingLearners(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(res_.server_entity_size(), 1);
  EXPECT_EQ(res_.server_entity(0).hostname(), "localhost");
  EXPECT_EQ(res_.server_entity(0).port(), 1991);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_EmptyRequest) {
  JoinFederationRequest req_;
  JoinFederationResponse res_;
  auto status = service_->JoinFederation(&ctx_, &req_, &res_);

  EXPECT_FALSE(status.ok());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_NewLearner) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_,
              AddLearner(EqualsProto(learner_state.server_entity()),
                         EqualsProto(learner_state.local_dataset_spec())))
      .Times(Exactly(1))
      .WillOnce(Return(learner_state));

  JoinFederationRequest req_;
  JoinFederationResponse res_;
  req_.set_learner_id(learner_state.learner_id());
  *req_.mutable_server_entity() = learner_state.server_entity();
  *req_.mutable_local_dataset_spec() = learner_state.local_dataset_spec();

  auto status = service_->JoinFederation(&ctx_, &req_, &res_);

  EXPECT_TRUE(status.ok());
  EXPECT_FALSE(res_.learner_id().empty());
  EXPECT_FALSE(res_.auth_token().empty());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_LearnerServiceUnreachable) {
  // TODO(canastas): Implement this after we have the service alive check functionality.
  bool is_learner_reachable = true;
  EXPECT_TRUE(is_learner_reachable);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, JoinFederation_LearnerCollision) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_,
              AddLearner(EqualsProto(learner_state.server_entity()),
                         EqualsProto(learner_state.local_dataset_spec())))
      .Times(Exactly(2))
      .WillOnce(Return(learner_state))
      .WillOnce(Return(absl::AlreadyExistsError("Learner has already joined.")));

  JoinFederationRequest req_;
  JoinFederationResponse res_;
  req_.set_learner_id(learner_state.learner_id());
  *req_.mutable_server_entity() = learner_state.server_entity();
  *req_.mutable_local_dataset_spec() = learner_state.local_dataset_spec();

  // First time, learner joins successfully.
  service_->JoinFederation(&ctx_, &req_, &res_);

  // Second time, must return and AlreadyExists error.
  auto status = service_->JoinFederation(&ctx_, &req_, &res_);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_EmptyRequest) {
  LeaveFederationRequest req_;
  LeaveFederationResponse res_;
  auto status = service_->LeaveFederation(&ctx_, &req_, &res_);

  EXPECT_FALSE(status.ok());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerExists) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_,RemoveLearner(learner_state.learner_id(), learner_state.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::OkStatus()));

  LeaveFederationRequest req;
  req.set_auth_token(learner_state.auth_token());
  req.set_learner_id(learner_state.learner_id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(res.ack().status());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerNotExists) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_,RemoveLearner(learner_state.learner_id(), learner_state.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::NotFoundError("No such learner.")));

  LeaveFederationRequest req;
  req.set_auth_token(learner_state.auth_token());
  req.set_learner_id(learner_state.learner_id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

// NOLINTNEXTLINE
TEST_F(ControllerServicerImplTest, LeaveFederation_LearnerInvalidCredentials) {
  auto learner_state = ParseTextOrDie<LearnerState>(kLearnerState);

  EXPECT_CALL(controller_,RemoveLearner(learner_state.learner_id(), learner_state.auth_token()))
      .Times(Exactly(1))
      .WillOnce(Return(absl::UnauthenticatedError("Incorrect token.")));

  LeaveFederationRequest req;
  req.set_auth_token(learner_state.auth_token());
  req.set_learner_id(learner_state.learner_id());
  LeaveFederationResponse res;

  auto status = service_->LeaveFederation(&ctx_, &req, &res);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(res.ack().status());
}

} // namespace
} // namespace projectmetis::controller