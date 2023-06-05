
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_AGGREGATION_FUNCTION_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_AGGREGATION_FUNCTION_H_

#include <vector>
#include <utility>

#include "projectmetis/proto/model.pb.h"

namespace projectmetis::controller {

// Given that we need to define an interface, we basically need to provide the
// signature of pure virtual functions Recall that, a pure virtual function is
// a function that has to be assigned the value of 0.
class AggregationFunction {
 public:
  virtual ~AggregationFunction() = default;

  // Compute the community model given a sequence of federated models and their
  // respective scaling factors. Each federated model refers to the local
  // model of a learner and the associated scaling factor value refers to the
  // contribution value of the local model in the federation.
  virtual FederatedModel Aggregate(std::vector<std::pair<const Model*, double>>& pairs) = 0;

  virtual std::string name() = 0;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_AGGREGATION_FUNCTION_H_
