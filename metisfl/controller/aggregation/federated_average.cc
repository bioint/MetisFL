
#include <omp.h>

#include "metisfl/controller/aggregation/federated_average.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

namespace projectmetis::controller {
namespace {

using ::proto::DeserializeTensor;
using ::proto::SerializeTensor;

template<typename T>
void AddTensors(std::vector<T> &tensor_left,
                const TensorSpec &tensor_spec_right,
                double scaling_factor_right) {

  /**
   * The function first deserializes both tensors based on the provided data type. Then it
   * scales the right-hand-side tensor using its given scaling factor and then adds the
   * scaled right-hand-side tensor to the left-hand-side tensor. Finally, it serialized
   * the aggregated tensor and returns its string representation.
   */
  auto t2_r = DeserializeTensor<T>(tensor_spec_right);

  // Scale the right tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, scaling_factor_right));

  // Add (plus) the scaled right tensor to the left tensor. Addition occurs based on data type T.
  transform(tensor_left.begin(), tensor_left.end(), t2_r.begin(), tensor_left.begin(), std::plus<T>());

}

template<typename T>
std::vector<T> AggregateTensorAtIndex(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs,
    int var_idx,
    uint32_t var_num_values) {

  auto aggregated_tensor = std::vector<T>(var_num_values);
  for (const auto &pair: pairs) {
    const auto *local_model = pair.front().first;
    const double local_model_contrib_value = pair.front().second;
    const auto &local_variable = local_model->variables(var_idx);
    if (local_variable.has_plaintext_tensor()) {
      AddTensors(aggregated_tensor,
                 local_variable.plaintext_tensor().tensor_spec(),
                 local_model_contrib_value);
    } else {
      throw std::runtime_error("Unsupported variable type.");
    }
  }
  return aggregated_tensor;
}

}

/*
 * To aggregate models passed through the pairs list using Federated Average,
 * we assume that the second value (the contribution value or model scaling factor)
 * of each pair item in the pairs list is already scaled / normalized.
 * @pairs Represents the container of <Model Proto, Model Contribution Value(scaled)>
 * holding the local models over which we want to compute the aggregated model.
 */
FederatedModel
FederatedAverage::Aggregate(
    std::vector<std::vector<std::pair<const Model *, double>>> &pairs) {

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
  for (const auto &sample_variable: sample_model->variables()) {
    auto *variable = global_model.mutable_model()->add_variables();
    variable->set_name(sample_variable.name());
    variable->set_trainable(sample_variable.trainable());
    if (sample_variable.has_plaintext_tensor()) {
      *variable->mutable_plaintext_tensor()->mutable_tensor_spec() =
          sample_variable.plaintext_tensor().tensor_spec();
      *variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value() = "";
    } else {
      throw std::runtime_error("Only Plaintext variables are supported.");
    }
  }

  // TODO(dstripelis) We need to add support to aggregate only the trainable
  //  weights. For now, we aggregate all matrices, but if we aggregate only the
  //  trainable, then what should be the value of the non-trainable weights?
  auto total_variables = global_model.model().variables_size();
  #pragma omp parallel for
  for (int var_idx = 0; var_idx < total_variables; ++var_idx) {
      auto var_data_type = global_model.model().variables(var_idx).plaintext_tensor().tensor_spec().type().type();
      auto var_num_values = global_model.model().variables(var_idx).plaintext_tensor().tensor_spec().length();

      std::vector<char> serialized_tensor;
      if (var_data_type == DType_Type_UINT8) {
        auto aggregated_tensor = AggregateTensorAtIndex<unsigned char>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<unsigned char>(aggregated_tensor);
      } else if (var_data_type == DType_Type_UINT16) {
        auto aggregated_tensor = AggregateTensorAtIndex<unsigned short>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<unsigned short>(aggregated_tensor);
      } else if (var_data_type == DType_Type_UINT32) {
        auto aggregated_tensor = AggregateTensorAtIndex<unsigned int>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<unsigned int>(aggregated_tensor);
      } else if (var_data_type == DType_Type_UINT64) {
        auto aggregated_tensor = AggregateTensorAtIndex<unsigned long>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<unsigned long>(aggregated_tensor);
      } else if (var_data_type == DType_Type_INT8) {
        auto aggregated_tensor = AggregateTensorAtIndex<signed char>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<signed char>(aggregated_tensor);
      } else if (var_data_type == DType_Type_INT16) {
        auto aggregated_tensor = AggregateTensorAtIndex<signed short>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<signed short>(aggregated_tensor);
      } else if (var_data_type == DType_Type_INT32) {
        auto aggregated_tensor = AggregateTensorAtIndex<signed int>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<signed int>(aggregated_tensor);
      } else if (var_data_type == DType_Type_INT64) {
        auto aggregated_tensor = AggregateTensorAtIndex<signed long>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<signed long>(aggregated_tensor);
      } else if (var_data_type == DType_Type_FLOAT32) {
        auto aggregated_tensor = AggregateTensorAtIndex<float>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<float>(aggregated_tensor);
      } else if (var_data_type == DType_Type_FLOAT64) {
        auto aggregated_tensor = AggregateTensorAtIndex<double>(pairs, var_idx, var_num_values);
        serialized_tensor = SerializeTensor<double>(aggregated_tensor);
      } else {
        throw std::runtime_error("Unsupported tensor data type.");
      }
      // Convert the char vector representing the aggregated result to string and assign to variable.
      std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
      *global_model.mutable_model()->mutable_variables(var_idx)->
          mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value() =
              serialized_tensor_str;
  }

  // Sets the number of contributors to the number of input models.
  global_model.set_num_contributors(pairs.size());
  return global_model;

}

void FederatedAverage::Reset() {
  // pass
}

} // namespace projectmetis::controller
