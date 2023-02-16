
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_

#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "projectmetis/proto/controller.grpc.pb.h"

namespace projectmetis::controller {

// TODO(stripeli): Add a nice description about what the controller is about :)
class Controller {
 public:
  virtual ~Controller() = default;

  // Returns the parameters with which the controller was initialized.
  ABSL_MUST_USE_RESULT
  virtual const ControllerParams &GetParams() const = 0;

  // Returns the list of all the active learners.
  ABSL_MUST_USE_RESULT
  virtual std::vector<LearnerDescriptor> GetLearners() const = 0;

  // Returns the number of active learners.
  ABSL_MUST_USE_RESULT
  virtual uint32_t GetNumLearners() const = 0;

  // Returns the current community model;
  ABSL_MUST_USE_RESULT
  virtual const FederatedModel& CommunityModel() const = 0;

  // Overwrites/replaces the community model with the provided.
  virtual absl::Status ReplaceCommunityModel(const FederatedModel& model) = 0;

  // Attempts to add a new learner to the federation. If successful, a
  // LearnerState instance is returned. Otherwise, it returns null.
  virtual absl::StatusOr<LearnerDescriptor>
  AddLearner(const ServerEntity &server_entity,
             const DatasetSpec &dataset_spec) = 0;

  // Removes a learner from the federation. If successful it returns OK.
  // Otherwise, it returns an error.
  virtual absl::Status
  RemoveLearner(const std::string &learner_id, const std::string &token) = 0;

  virtual absl::Status
  LearnerCompletedTask(const std::string &learner_id,
                       const std::string &token,
                       const CompletedLearningTask &task) = 0;

  virtual std::vector<FederatedTaskRuntimeMetadata>
  GetRuntimeMetadataLineage(uint32_t num_steps) = 0;

  virtual std::vector<CommunityModelEvaluation>
  GetEvaluationLineage(uint32_t num_steps) = 0;

  virtual std::vector<TaskExecutionMetadata>
  GetLocalTaskLineage(const std::string &learner_id, uint32_t num_steps) = 0;

  virtual void Shutdown() = 0;

public:
  // Creates a new controller using the default implementation, i.e., in-memory.
  static std::unique_ptr<Controller> New(const ControllerParams &params);
};

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_H_
