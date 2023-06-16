
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_

#include <vector>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "metisfl/proto/metis.pb.h"

namespace projectmetis::controller {

// Given that we need to define an interface, we basically need to provide the
// signature of pure virtual functions Recall that, a pure virtual function is
// a function that has to be assigned the value of 0.
class ScalingFunction {
 public:
  virtual ~ScalingFunction() = default;

  // The argument community_model refers to the latest community/global model.
  // The argument all_states refers to all the learners currently available in the federation.
  // The argument participating_states refers to all the learners used for aggregation.
  // The argument participating_metadata refers to the metadata of the learners used for aggregation.
  virtual absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState> &all_states,
      const absl::flat_hash_map<std::string, LearnerState*> &participating_states,
      const absl::flat_hash_map<std::string, TaskExecutionMetadata*> &participating_metadata) = 0;

  virtual std::string Name() = 0;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_
