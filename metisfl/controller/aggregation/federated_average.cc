
#include "metisfl/controller/aggregation/federated_average.h"

using metisfl::proto;

namespace metisfl::controller {
namespace {

template <typename T>
void AddTensors(std::vector<T> &tensor_left, const Tensor &tensor_spec_right,
                double scaling_factor_right) {
  /**
   * The function first deserializes both tensors based on the provided data
   * type. Then it scales the right-hand-side tensor using its given scaling
   * factor and then adds the scaled right-hand-side tensor to the
   * left-hand-side tensor. Finally, it serialized the aggregated tensor and
   * returns its string representation.
   */
  auto t2_r = ProtoSerde::DeserializeTensor<T>(tensor_spec_right);

  // Scale the right tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1,
                      scaling_factor_right));

  // Add (plus) the scaled right tensor to the left tensor. Addition occurs
  // based on data type T.
  transform(tensor_left.begin(), tensor_left.end(), t2_r.begin(),
            tensor_left.begin(), std::plus<T>());
}

template <typename T>
std::vector<T> AggregateTensorAtIndex(AggregationPairs &pairs, int var_idx,
                                      uint32_t var_num_values) {
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

}  // namespace

/*
 * To aggregate models passed through the pairs list using Federated Average,
 * we assume that the second value (the contribution value or model scaling
 * factor) of each pair item in the pairs list is already scaled / normalized.
 * @pairs Represents the container of <Model Proto, Model Contribution
 * Value(scaled)> holding the local models over which we want to compute the
 * aggregated model.
 */
template <typename T>
FederatedModel FederatedAverage::Aggregate(AggregationPairs &pairs) {
  // With the new global model aggregation function
  // NEW API GUIDE
  // { {(*model, 0.2)}, {(*model, 0.6)}, {(*model, 0.5)}}
  // pairs[0] -> {(*model, 0.2)}
  // pairs[0][0] ->  (*model, 0.2)
  // pairs[0][0].first -> *model
  // pairs[0][0].second -> 0.2
  // Initializes an empty global model.
  FederatedModel global_model;
  const auto &sample_model = pairs.front().front().first;

  global_model.mutable_model()->mutable_tensors()->CopyFrom(
      sample_model->tensors());

  auto total_tensors = global_model.model().tensors_size();

#pragma omp parallel for
  for (int var_idx = 0; var_idx < total_tensors; ++var_idx) {
    auto var_data_type = global_model.model().tensors(var_idx).type().type();
    auto var_num_values = global_model.model().tensors(var_idx).length();

    auto aggregated_tensor =
        AggregateTensorAtIndex<T>(pairs, var_idx, var_num_values);
    auto serialized_tensor = ProtoSerde::SerializeTensor<T>(aggregated_tensor);

    std::string serialized_tensor_str(serialized_tensor.begin(),
                                      serialized_tensor.end());

    *global_model.mutable_model()->mutable_tensors(var_idx)->mutable_value() =
        serialized_tensor_str;
  }

  global_model.set_num_contributors(pairs.size());
  return global_model;
}

void FederatedAverage::Reset() {}

}  // namespace metisfl::controller
