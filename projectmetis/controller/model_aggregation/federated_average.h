
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FEDERATED_AVERAGE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FEDERATED_AVERAGE_H_

#include "projectmetis/controller/model_aggregation/aggregation_function.h"
#include "projectmetis/proto/model.pb.h"

namespace projectmetis::controller {

class FederatedAverage : public AggregationFunction {
 public:
  FederatedModel Aggregate(std::vector<std::pair<const Model*, double>>& pairs) override;

  inline std::string name() override {
    return "FedAvg";
  }
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_AGGREGATIONS_FEDERATED_AVERAGE_H_
