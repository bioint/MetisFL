
#include "metisfl/controller/aggregation/federated_average.h"

namespace metisfl::controller {

Model FederatedAverage::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {
  const auto &sample_model = pairs.front().front().first;
  if (sample_model->encrypted()) {
    throw std::runtime_error(
        "Cannot aggregate encrypted tensors using "
        "Federated Average.");
  }

  Model model;
  model.mutable_tensors()->CopyFrom(sample_model->tensors());

  auto total_tensors = model.tensors_size();

#pragma omp parallel for
  for (int var_idx = 0; var_idx < total_tensors; ++var_idx) {
    auto var_num_values = model.tensors(var_idx).length();

    auto aggregated_tensor =
        AggregateTensorAtIndex(pairs, var_idx, var_num_values);
    auto serialized_tensor = TensorOps::SerializeTensor(aggregated_tensor);

    std::string serialized_tensor_str(serialized_tensor.begin(),
                                      serialized_tensor.end());

    *model.mutable_tensors(var_idx)->mutable_value() = serialized_tensor_str;
  }

  return model;
}

std::vector<double> FederatedAverage::AggregateTensorAtIndex(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs,
    int var_idx, uint32_t var_num_values) const {
  auto aggregated_tensor = std::vector<double>(var_num_values);
  for (const auto &pair : pairs) {
    const auto *local_model = pair.front().first;
    const auto &local_tensor = local_model->tensors(var_idx);
    AddTensors(aggregated_tensor, local_tensor, pair.front().second);
  }
  return aggregated_tensor;
}

void FederatedAverage::AddTensors(std::vector<double> &tensor_left,
                                  const Tensor &tensor_spec_right,
                                  double scaling_factor_right) const {
  auto t2_r = TensorOps::DeserializeTensor(tensor_spec_right);

  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1,
                      scaling_factor_right));

  transform(tensor_left.begin(), tensor_left.end(), t2_r.begin(),
            tensor_left.begin(), std::plus<double>());
}

void FederatedAverage::Reset() {}

}  // namespace metisfl::controller
