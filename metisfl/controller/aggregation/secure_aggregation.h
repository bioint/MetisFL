
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_

#include <omp.h>

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/encryption/encryption_scheme.h"
#include "metisfl/encryption/palisade/ckks_scheme.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

class SecAgg : AggregationFunction {
 public:
  SecAgg(const int batch_size, const int scaling_factor_bits,
         const std::string &crypto_context);

  FederatedModel Aggregate(std::vector<std::vector<std::pair<Model *, double>>> &pairs) override;

  [[nodiscard]] inline std::string Name() const override { return "SecAgg"; }

  [[nodiscard]] inline int RequiredLearnerLineageLength() const override {
    return 1;
  }

  void Reset() override;

 private:
  std::unique_ptr<EncryptionScheme> encryption_scheme_;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
