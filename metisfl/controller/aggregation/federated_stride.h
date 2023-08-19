
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_SYNC_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_SYNC_H_

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/controller/aggregation/federated_rolling_average_base.h"


namespace metisfl::controller {

class FederatedStride : public AggregationFunction, FederatedRollingAverageBase {

 public:
  FederatedModel Aggregate(std::vector<std::vector<std::pair<const Model *, double>>> &pairs) override;

  inline std::string Name() const override {
    return "FedStride";
  }

  inline int RequiredLearnerLineageLength() const override {
    return 1;
  }

  void Reset() override;

};

}

#endif //METISFL_METISFL_CONTROLLER_AGGREGATION_FED_ROLL_SYNC_H_
