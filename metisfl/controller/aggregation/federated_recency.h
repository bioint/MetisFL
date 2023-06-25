
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_

#include "metisfl/controller/aggregation/federated_rolling_average_base.h"
#include "metisfl/controller/aggregation/aggregation_function.h"

namespace metisfl::controller {

class FederatedRecency : public AggregationFunction, FederatedRollingAverageBase {

 public:
  FederatedModel Aggregate(std::vector<std::vector<std::pair<const Model *, double>>> &pairs) override;

  [[nodiscard]] inline std::string Name() const override {
    return "FedRec";
  }

  [[nodiscard]] inline int RequiredLearnerLineageLength() const override {
    return 2; // The Async strategy keeps upto two models in its cache.
  }

  void Reset() override;

};

}

#endif //METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_ASYNC_H_
