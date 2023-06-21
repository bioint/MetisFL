
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_

#include "metisfl/encryption/palisade/fhe_helper.h"
#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/proto/model.pb.h"
#include "metisfl/proto/metis.pb.h"

namespace projectmetis::controller {

class PWA : public AggregationFunction {
 private:
  HEScheme he_scheme_;
  FHE_Helper fhe_helper_;

 public:
  explicit PWA(const HEScheme &he_scheme);

  FederatedModel Aggregate(std::vector<std::vector<std::pair<const Model*, double>>>& pairs) override;

  [[nodiscard]] inline std::string Name() const override {
    return "PWA";
  }

  [[nodiscard]] inline int RequiredLearnerLineageLength() const override {
    return 1;
  }

  void Reset() override;

};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
