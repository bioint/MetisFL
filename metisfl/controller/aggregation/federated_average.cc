
#include "metisfl/controller/aggregation/federated_average.h"

namespace metisfl::controller {

template <typename T>
FederatedModel FederatedAverage<T>::Aggregate(
    std::vector<std::vector<std::pair<Model *, double>>> &pairs) {
  FederatedModel global_model;

  const auto &sample_model = pairs.front().front().first;

  global_model.mutable_model()->mutable_tensors()->CopyFrom(
      sample_model->tensors());

  auto total_tensors = global_model.model().tensors_size();

#pragma omp parallel for
  for (int var_idx = 0; var_idx < total_tensors; ++var_idx) {
    auto var_num_values = global_model.model().tensors(var_idx).length();

    auto aggregated_tensor =
        AggregateTensorAtIndex(pairs, var_idx, var_num_values);
    auto serialized_tensor = SerializeTensor<T>(aggregated_tensor);

    std::string serialized_tensor_str(serialized_tensor.begin(),
                                      serialized_tensor.end());

    *global_model.mutable_model()->mutable_tensors(var_idx)->mutable_value() =
        serialized_tensor_str;
  }

  return global_model;
}
template <typename T>
void FederatedAverage<T>::Reset() {}

template <typename T>
void FederatedAverage<T>::AddTensors(std::vector<T> &tensor_left,
                                     const Tensor &tensor_spec_right,
                                     double scaling_factor_right) const {
  auto t2_r = DeserializeTensor<T>(tensor_spec_right);

  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1,
                      scaling_factor_right));

  transform(tensor_left.begin(), tensor_left.end(), t2_r.begin(),
            tensor_left.begin(), std::plus<T>());
}

template <typename T>
std::vector<T> FederatedAverage<T>::AggregateTensorAtIndex(
    std::vector<std::vector<std::pair<Model *, double>>> &pairs, int var_idx,
    uint32_t var_num_values) const {
  auto aggregated_tensor = std::vector<T>(var_num_values);
  for (const auto &pair : pairs) {
    const auto *local_model = pair.front().first;
    const auto &local_tensor = local_model->tensors(var_idx);
    if (!local_tensor.encrypted()) {
      AddTensors(aggregated_tensor, local_tensor, pair.front().second);
    } else {
      throw std::runtime_error(
          "Cannot aggregate encrypted tensors using "
          "Federated Average.");
    }
  }
  return aggregated_tensor;
}

}  // namespace metisfl::controller
