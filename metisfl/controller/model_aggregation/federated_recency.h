
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_ASYNC_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_ASYNC_H_

#include "metisfl/controller/model_aggregation/federated_rolling_average_base.h"
#include "metisfl/controller/model_aggregation/aggregation_function.h"

namespace projectmetis::controller {

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

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_ASYNC_H_
