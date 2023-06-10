
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_

#include <gmock/gmock.h>

#include "metisfl/controller/controller.h"

namespace projectmetis::controller {

class MockController : public Controller {
 public:
  MOCK_METHOD(ControllerParams &, GetParams, (), (const, override));
  MOCK_METHOD(std::vector<LearnerDescriptor>,
              GetLearners,
              (),
              (const, override));
  MOCK_METHOD(uint32_t, GetNumLearners, (), (const, override));
  MOCK_METHOD(absl::StatusOr<LearnerDescriptor>,
              AddLearner,
              (const ServerEntity &server_entity, const DatasetSpec &dataset_spec),
              (override));
  MOCK_METHOD(absl::Status,
              RemoveLearner,
              (const std::string &learner_id, const std::string &token),
              (override));
  MOCK_METHOD(absl::Status,
              LearnerCompletedTask,
              (const std::string &learner_id, const std::string &token, const CompletedLearningTask &task),
              (override));
  MOCK_METHOD(std::vector<ModelEvaluation>,
              GetEvaluationLineage,
              (const std::string &learner_id, uint32_t num_steps),
              (override));
  MOCK_METHOD(std::vector<ModelEvaluation>,
              GetEvaluationLineage,
              (uint32_t num_steps),
              (override));
  MOCK_METHOD(FedRuntimeMetadata&,
              RuntimeMetadata,
              (),
              (const, override));
  MOCK_METHOD(absl::Status,
              ReplaceCommunityModel,
              (const FederatedModel& model),
              (override));
  MOCK_METHOD(FederatedModel&,
              CommunityModel,
              (),
              (const, override));
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_MOCK_H_
