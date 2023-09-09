
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_

#include <omp.h>

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

class SecAgg : public AggregationFunction {
 public:
  SecAgg(int batch_size, int scaling_factor_bits, std::string crypto_context);

  Model Aggregate(std::vector<std::vector<std::pair<const Model *, double>>>
                      &pairs) override;

  inline std::string Name() const override { return "SecAgg"; }

  inline int RequiredLearnerLineageLength() const override { return 1; }

  void Reset() override;

 private:
  std::unique_ptr<CKKS> ckks_;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
