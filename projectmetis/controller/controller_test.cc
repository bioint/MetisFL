
#include <memory>

#include <gtest/gtest.h>

#include "projectmetis/controller/controller.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {
namespace {

class ControllerTest : public ::testing::Test {
 public:

  static ControllerParams CreateDefaultParams() {
    // Construct default (testing) parameters to initialize controller.
    ControllerParams params;

    // Set controller server connection parameters.
    params.mutable_server_entity()->set_hostname("0.0.0.0");
    params.mutable_server_entity()->set_port(50051);

    // Set federated training protocol specifications.
    params.mutable_global_model_specs()
        ->set_learners_participation_ratio(1);
    params.mutable_global_model_specs()->set_aggregation_rule(
        GlobalModelSpecs::FED_AVG);
    params.mutable_communication_specs()->set_protocol(
        CommunicationSpecs::SYNCHRONOUS);

    // Set Fully Homomorphic Encryption specifications.
    *params.mutable_fhe_scheme() = FHEScheme();

    // Set model store specifications.
    ModelStoreConfig model_store_config;
    *model_store_config.mutable_in_memory_store() = InMemoryStore();
    *params.mutable_model_store_config() = model_store_config;

    // Set model hyperparams.
    params.mutable_model_hyperparams()->set_epochs(10);
    params.mutable_model_hyperparams()->set_batch_size(100);
    params.mutable_model_hyperparams()->set_percent_validation(5);

    return params;
  }

  static std::unique_ptr<Controller> CreateEmptyController() {
    auto default_params = CreateDefaultParams();
    // A controller with no registered learner.
    return Controller::New(default_params);
  }

  static std::unique_ptr<Controller> CreateController() {
    auto default_params = CreateDefaultParams();
    // A controller with a single registered learner.
    auto controller = Controller::New(default_params);

    auto learner = ServerEntity();
    learner.set_hostname("localhost");
    learner.set_port(50052);

    auto dataset = DatasetSpec();
    dataset.set_num_training_examples(1);
    dataset.set_num_validation_examples(1);
    dataset.set_num_test_examples(1);

    EXPECT_TRUE(controller->AddLearner(learner, dataset).ok());

    return controller;
  }
};

TEST_F(ControllerTest, GetParamsNotEmpty) /* NOLINT */ {
  auto controller = CreateEmptyController();
  const auto params = controller->GetParams();
  EXPECT_TRUE(params.IsInitialized());
  controller->Shutdown();
}

TEST_F(ControllerTest, GetLearnersEmpty) /* NOLINT */ {
  auto controller = CreateEmptyController();
  const auto learners = controller->GetLearners();
  EXPECT_TRUE(learners.empty());
  controller->Shutdown();
}

TEST_F(ControllerTest, GetLearnersNotEmpty) /* NOLINT */ {
  auto controller = CreateController();
  const auto learners = controller->GetLearners();
  EXPECT_FALSE(learners.empty());
  controller->Shutdown();
}

//TEST_F(ControllerTest, AddLearnerNewEntity) /* NOLINT */ {
//  auto controller = CreateEmptyController();
//
//  auto learner = ServerEntity();
//  learner.set_hostname("localhost");
//  learner.set_port(1991);
//
//  auto dataset = DatasetSpec();
//  dataset.set_num_training_examples(1);
//  dataset.set_num_validation_examples(1);
//  dataset.set_num_test_examples(1);
//
//  auto status = controller->AddLearner(learner, dataset);
//
//  EXPECT_TRUE(status.ok());
//}
//
//TEST_F(ControllerTest, AddLearnerExistingEntity) /* NOLINT */ {
//  auto controller = CreateController();
//
//  auto learner = ServerEntity();
//  learner.set_hostname("localhost");
//  learner.set_port(1991);
//
//  auto dataset = DatasetSpec();
//  dataset.set_num_training_examples(1);
//  dataset.set_num_validation_examples(1);
//  dataset.set_num_test_examples(1);
//
//  auto status = controller->AddLearner(learner, dataset);
//
//  EXPECT_FALSE(status.ok());
//}
//
//TEST_F(ControllerTest, AddLearnerEmptyFields) /* NOLINT */ {
//  auto controller = CreateController();
//
//  auto status = controller->AddLearner(ServerEntity(), DatasetSpec());
//  EXPECT_FALSE(status.ok());
//}
//
//TEST_F(ControllerTest, RemoveLearnerExistingEntity) /* NOLINT */ {
//  auto controller = CreateController();
//  auto learners = controller->GetLearners();
//
//  // We already know that the Controller is initialized with a single learner
//  // therefore the 0 index on the returned vector.
//  auto status = controller->RemoveLearner(learners[0].id(),
//                                          learners[0].auth_token());
//  EXPECT_TRUE(status.ok());
//}
//
//TEST_F(ControllerTest, RemoveLearnerExistingEntityWrongToken) /* NOLINT */ {
//  auto controller = CreateController();
//  auto learners = controller->GetLearners();
//
//  // We already know that the Controller is initialized with a single learner
//  // therefore the 0 index on the returned vector.
//  auto status = controller->RemoveLearner(learners[0].id(),
//                                          "foobar");
//  EXPECT_FALSE(status.ok());
//}
//
//TEST_F(ControllerTest, RemoveLearnerNotExistingEntity) /* NOLINT */ {
//  auto controller = CreateEmptyController();
//
//  auto status = controller->RemoveLearner("foo", "bar");
//  EXPECT_FALSE(status.ok());
//}
//
//TEST_F(ControllerTest, RemoveLearnerEmptyFields) /* NOLINT */ {
//  auto controller = CreateEmptyController();
//
//  auto status = controller->RemoveLearner(std::string(), std::string());
//  EXPECT_FALSE(status.ok());
//
//  status = controller->RemoveLearner("foobar", std::string());
//  EXPECT_FALSE(status.ok());
//
//  status = controller->RemoveLearner(std::string(), "foobar");
//  EXPECT_FALSE(status.ok());
//}

} // namespace
} // namespace projectmetis::controller
