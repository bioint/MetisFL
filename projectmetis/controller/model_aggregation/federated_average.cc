
#include "projectmetis/controller/model_aggregation/federated_average.h"
#include "projectmetis/core/proto_tensor_serde.h"
#include "projectmetis/proto/model.pb.h"

namespace projectmetis::controller {
namespace {

template<typename T>
std::string AddTensors(const TensorSpec &tensor_spec_left,
                       const TensorSpec &tensor_spec_right,
                       double scaling_factor_right) {

  /**
   * The function first deserializes both tensors based on the provided data type. Then it
   * scales the right-hand-side tensor using its given scaling factor and then adds the
   * scaled right-hand-side tensor to the left-hand-side tensor. Finally, it serialized
   * the aggregated tensor and returns its string representation.
   */
  auto t1_l = ::proto::DeserializeTensor<T>(tensor_spec_left);
  auto t2_r = ::proto::DeserializeTensor<T>(tensor_spec_right);

  // Scale the right tensor by its scaling factor.
  // Careful here: if the data type is uint or int then there are no precision
  // bits and therefore the number will be rounded to the smallest integer.
  // For instance, if 0.5 * 3 then the result is 1, int(0.5*3) = int(1.5) = 1
  transform(t2_r.begin(), t2_r.end(), t2_r.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, scaling_factor_right));

  // Add (plus) the scaled right tensor to the left tensor. Addition occurs based on data type T.
  transform(t1_l.begin(), t1_l.end(), t2_r.begin(), t1_l.begin(), std::plus<T>());

  // Serialize aggregated result.
  auto serialized_tensor = ::proto::SerializeTensor<T>(t1_l);
  // Convert serialization to string.
  std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
  return serialized_tensor_str;

}

std::string AddTensors(const TensorSpec &tensor_spec_left,
                       const TensorSpec &tensor_spec_right,
                       double scaling_factor_right) {

  /**
   * This is basically a wrapper over the AddTensors function. It calls the AddTensors
   * function by first casting it to the given data type. Then returns the aggregated tensor.
   */
  auto num_values_left = tensor_spec_left.length();
  auto num_values_right = tensor_spec_right.length();

  auto data_type_left = tensor_spec_left.type().type();
  auto data_type_right = tensor_spec_right.type().type();

  if (num_values_left != num_values_right) throw std::runtime_error("Left and right tensors have different sizes");
  if (data_type_left != data_type_right) throw std::runtime_error("Left and right tensors have different data types");

  std::string aggregated_result;
  if (data_type_left == DType_Type_UINT8) {
    aggregated_result = AddTensors<unsigned char>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_UINT16) {
    aggregated_result = AddTensors<unsigned short>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_UINT32) {
    aggregated_result = AddTensors<unsigned int>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_UINT64) {
    aggregated_result = AddTensors<unsigned long>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_INT8) {
    aggregated_result = AddTensors<signed char>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_INT16) {
    aggregated_result = AddTensors<signed short>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_INT32) {
    aggregated_result = AddTensors<signed int>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_INT64) {
    aggregated_result = AddTensors<signed long>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_FLOAT32) {
    aggregated_result = AddTensors<float>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else if (data_type_left == DType_Type_FLOAT64) {
    aggregated_result = AddTensors<double>(tensor_spec_left, tensor_spec_right, scaling_factor_right);
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }

  return aggregated_result;

}

std::vector<char> GenSerializedEmptyTensor(const TensorSpec &tensor_spec) {

  /**
   * Creates a tensor of the given size and data type with zero values.
   * The serializes the tensor and converts its serialized version to string.
   */
  std::vector<char> serialized_tensor;
  auto num_values = tensor_spec.length();
  auto data_type = tensor_spec.type().type();

  if (data_type == DType_Type_UINT8) {
    serialized_tensor = ::proto::SerializeTensor<unsigned char>(std::vector<unsigned char>(num_values));
  } else if (data_type == DType_Type_UINT16) {
    serialized_tensor = ::proto::SerializeTensor<unsigned short>(std::vector<unsigned short>(num_values));
  } else if (data_type == DType_Type_UINT32) {
    serialized_tensor = ::proto::SerializeTensor<unsigned int>(std::vector<unsigned int>(num_values));
  } else if (data_type == DType_Type_UINT64) {
    serialized_tensor = ::proto::SerializeTensor<unsigned long>(std::vector<unsigned long>(num_values));
  } else if (data_type == DType_Type_INT8) {
    serialized_tensor = ::proto::SerializeTensor<signed char>(std::vector<signed char>(num_values));
  } else if (data_type == DType_Type_INT16) {
    serialized_tensor = ::proto::SerializeTensor<signed short>(std::vector<signed short>(num_values));
  } else if (data_type == DType_Type_INT32) {
    serialized_tensor = ::proto::SerializeTensor<signed int>(std::vector<signed int>(num_values));
  } else if (data_type == DType_Type_INT64) {
    serialized_tensor = ::proto::SerializeTensor<signed long>(std::vector<signed long>(num_values));
  } else if (data_type == DType_Type_FLOAT32) {
    serialized_tensor = ::proto::SerializeTensor<float>(std::vector<float>(num_values));
  } else if (data_type == DType_Type_FLOAT64) {
    serialized_tensor = ::proto::SerializeTensor<double>(std::vector<double>(num_values));
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }
  return serialized_tensor;

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
    std::vector<std::pair<const Model *, double>> &pairs) {

  // Initializes the community model to zero.
  FederatedModel global_model;
  const auto &sample_model = pairs.front().first;
  for (const auto &sample_variable: sample_model->variables()) {
    auto *variable = global_model.mutable_model()->add_variables();
    variable->set_name(sample_variable.name());
    variable->set_trainable(sample_variable.trainable());
    if (sample_variable.has_plaintext_tensor()) {
      *variable->mutable_plaintext_tensor()->mutable_tensor_spec() =
          sample_variable.plaintext_tensor().tensor_spec();
      auto serialized_tensor = GenSerializedEmptyTensor(sample_variable.plaintext_tensor().tensor_spec());
      std::string serialized_tensor_str(serialized_tensor.begin(), serialized_tensor.end());
      *variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value() = serialized_tensor_str;
    } else {
      throw std::runtime_error("Unsupported variable type.");
    }
  }

  // TODO(dstripelis) We need to add support to aggregate only the trainable
  //  weights. For now, we aggregate all matrices, but if we aggregate only the
  //  trainable, then what should be the value of the non-trainable weights?
  // Aggregates the given models.
  for (const auto &pair: pairs) {
    const auto *local_model = pair.first;
    const double local_model_contrib_value = pair.second;
    for (int i = 0; i < local_model->variables_size(); ++i) {
      const auto &local_variable = local_model->variables(i);
      auto global_variable = global_model.mutable_model()->mutable_variables(i);
      if (local_variable.has_plaintext_tensor()) {
        auto aggregated_result =
            AddTensors(global_variable->plaintext_tensor().tensor_spec(),
                       local_variable.plaintext_tensor().tensor_spec(),
                       local_model_contrib_value);
        // Assigns aggregated result to the global tensor.
        *global_variable->mutable_plaintext_tensor()->mutable_tensor_spec()->mutable_value() = aggregated_result;
      } else {
        throw std::runtime_error("Unsupported variable type.");
      }
    }
  }

  // Sets the number of contributors to the number of input models.
  global_model.set_num_contributors(pairs.size());
  return global_model;

}

} // namespace projectmetis::controller
