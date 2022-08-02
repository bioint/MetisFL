
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_

#include <vector>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

// Given that we need to define an interface, we basically need to provide the
// signature of pure virtual functions Recall that, a pure virtual function is
// a function that has to be assigned the value of 0.
class ScalingFunction {
 public:
  virtual ~ScalingFunction() = default;

  virtual absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const FederatedModel &community_model,
      const absl::flat_hash_map<std::string, LearnerState> &states) = 0;

  virtual std::string name() = 0;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SCALING_SCALING_FUNCTION_H_
