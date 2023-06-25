
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_

#include <vector>
#include <utility>

#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

// Given that we need to define an interface, we basically need to provide the
// signature of pure virtual functions Recall that, a pure virtual function is
// a function that has to be assigned the value of 0.
class AggregationFunction {
 public:
  virtual ~AggregationFunction() = default;

  // Compute the community model given a sequence of federated models and their
  // respective scaling factors. The function input has the following format:
  // [
  //    [(Model_A_1, Weight_A_1), (Model_A_2, Weight_A_2)],
  //    [(Model_B_1, Weight_B_1), (Model_B_2, Weight_B_2)],
  //    ...
  // ]
  // Specifically, a list of lists. Where the inner list holds the models
  // of each learner and their associated scaling factor values, which is the
  // contribution value of the model in the global/community model. The outer
  // list holds the list of models of every other learner.
  virtual FederatedModel Aggregate(std::vector<std::vector<std::pair<const Model*, double>>>& pairs) = 0;

  // Keeps track of the number of models per learner the current aggregation will operate on.
  [[nodiscard]] inline virtual int RequiredLearnerLineageLength() const = 0;

  [[nodiscard]] inline virtual std::string Name() const = 0;

  virtual void Reset() = 0;
};

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_
