
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_

#include "encryption/palisade/fhe/fhe_helper.h"

#include "projectmetis/controller/model_aggregation/aggregation_function.h"
#include "projectmetis/proto/model.pb.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

class PWA : public AggregationFunction {
 public:
  explicit PWA(const FHEScheme &fhe_scheme) :
  fhe_scheme_(fhe_scheme),
  fhe_helper_(fhe_scheme.name(), fhe_scheme.batch_size(), fhe_scheme.scaling_bits()) {
    fhe_helper_.load_crypto_params();
  }
  explicit PWA(FHEScheme &&fhe_scheme) :
  fhe_scheme_(std::move(fhe_scheme)),
  fhe_helper_(fhe_scheme.name(), fhe_scheme.batch_size(), fhe_scheme.scaling_bits()) {
    fhe_helper_.load_crypto_params();
  }
  FederatedModel Aggregate(std::vector<std::pair<const Model*, double>>& pairs) override;

  inline std::string name() override {
    return "PWA";
  }

 private:
  FHEScheme fhe_scheme_;
  FHE_Helper fhe_helper_;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_AGGREGATION_PRIVATE_WEIGHTED_AVERAGE_H_
