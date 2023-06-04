
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_SYNC_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_SYNC_H_

#include "src/cc/controller/model_aggregation/aggregation_function.h"
#include "src/cc/controller/model_aggregation/federated_rolling_average_base.h"
#include "src/proto/metis.pb.h"

namespace projectmetis::controller {

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

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FED_ROLL_SYNC_H_
