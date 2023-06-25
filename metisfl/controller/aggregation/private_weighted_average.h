
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"
#include "metisfl/proto/model.pb.h"
#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {

class PWA : public AggregationFunction {
 private:
  HESchemeConfig he_scheme_config_;
  CKKS he_scheme_;

 public:
  explicit PWA(const HESchemeConfig &he_scheme_config);

  FederatedModel Aggregate(std::vector<std::vector<std::pair<const Model*, double>>>& pairs) override;

  [[nodiscard]] inline std::string Name() const override {
    return "PWA";
  }

  [[nodiscard]] inline int RequiredLearnerLineageLength() const override {
    return 1;
  }

  void Reset() override;

};

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
