
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/controller/aggregation/federated_rolling_average_base.h"

namespace metisfl::controller {

template <typename T>
class FederatedRecency : public AggregationFunction,
                         FederatedRollingAverageBase<T> {
 public:
  Model Aggregate(
      std::vector<std::vector<std::pair<Model *, double>>> &pairs) override;

  inline std::string Name() const override { return "FedRec"; }

  inline int RequiredLearnerLineageLength() const override {
    return 2;  // The Async strategy keeps upto two models in its cache.
  }

  void Reset() override;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_
