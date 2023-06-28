
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_

#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "metisfl/proto/controller.grpc.pb.h"

namespace metisfl::controller {

/**
* The controller orchestrates the execution of the entire federated
* learning environment. It comprises of the following core procedures
* (with order of execution during federated traininig):
* - Training Task Scheduling: select the learners that will participate
*       in the training of the federated model, specify synchronization
*        points, dispatch the model training task to the selected learners,
*        and receive local models.
* - Model Storing: save the local models and the contribution value of
*        each learner to improve model aggregation efficiency in the
*        presence of many learners and/or large models.
* - Model Aggregation: mix/merge the local models of the earners and compute
*        a new global model. The aggregation can also take place in an
*        encrypted space (e.g. using homomorphic encryption).
* - Evaluation Task Scheduling: dispatch the global model evaluation task
*        to the learners and await to collect the respective metrics.
*/

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

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_H_
