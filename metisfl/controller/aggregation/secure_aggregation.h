
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/encryption/encryption_scheme.h"
#include "metisfl/proto/model.pb.h"
#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {

class SecAgg : public AggregationFunction {
 private:
  EncryptionConfig encryption_config_;
  std::unique_ptr<EncryptionScheme> encryption_scheme_;

 public:
  explicit SecAgg(const EncryptionConfig &encryption_config);

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

#endif //METISFL_METISFL_CONTROLLER_AGGREGATION_SECURE_AGGREGATION_H_
