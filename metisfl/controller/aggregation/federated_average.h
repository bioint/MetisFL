
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_FEDERATED_AVERAGE_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_FEDERATED_AVERAGE_H_

#include "metisfl/controller/aggregation/aggregation_function.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

using metisfl::proto::DeserializeTensor;
using metisfl::proto::SerializeTensor;

namespace metisfl::controller {

template <typename T>
class FederatedAverage : public AggregationFunction {
 public:
  Model Aggregate(
      std::vector<std::vector<std::pair<Model *, double>>> &pairs) override;

  inline std::string Name() const override { return "FedAvg"; }

  inline int RequiredLearnerLineageLength() const override { return 1; }

  void Reset() override;

 private:
  void AddTensors(std::vector<T> &tensor_left, const Tensor &tensor_spec_right,
                  double scaling_factor_right) const;

  std::vector<T> AggregateTensorAtIndex(
      std::vector<std::vector<std::pair<Model *, double>>> &pairs, int var_idx,
      uint32_t var_num_values) const;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_FEDERATED_AVERAGE_H_
