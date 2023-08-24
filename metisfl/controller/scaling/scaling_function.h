
#ifndef METISFL_METISFL_CONTROLLER_SCALING_SCALING_FUNCTION_H_
#define METISFL_METISFL_CONTROLLER_SCALING_SCALING_FUNCTION_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "metisfl/proto/controller.pb.h"

namespace metisfl::controller {

// Given that we need to define an interface, we basically need to provide the
// signature of pure virtual functions Recall that, a pure virtual function is
// a function that has to be assigned the value of 0.
class ScalingFunction {
 public:
  virtual ~ScalingFunction() = default;

  virtual absl::flat_hash_map<std::string, double> ComputeScalingFactors(
      const absl::flat_hash_map<std::string, Learner> &all_states,
      const absl::flat_hash_map<std::string, Learner *>
          &participating_states,
      const absl::flat_hash_map<std::string, TrainingMetadata *>
          &participating_metadata) = 0;

  virtual std::string Name() = 0;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCALING_SCALING_FUNCTION_H_
