
#ifndef METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
#define METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_

#include "metisfl/proto/model.pb.h"

#include <cstring>
#include <vector>

namespace proto {
namespace {

template<typename T>
inline std::vector<T> DeserializeTensor(const metisfl::TensorSpec &tensor_spec) {
  const auto tensor_bytes = tensor_spec.value().c_str();
  const auto tensor_elements_num = tensor_spec.length();
  std::vector<T> deserialized_tensor(tensor_elements_num);
  // Memory copy (memcpy) signature: std::memcpy(dest, src, count) where count
  // is the number of bytes to copy from src to dest.
  // CAUTION: we start from index 0 for the loaded_values vector
  // because we want to start appending to it the transformed values.
  std::memcpy(&deserialized_tensor[0], tensor_bytes, tensor_elements_num * sizeof(T));
  return deserialized_tensor;
}

template<typename T>
inline std::vector<char> SerializeTensor(const std::vector<T> &v) {
  auto num_elements = v.size();
  std::vector<char> serialized_tensor(num_elements * sizeof(T));
  std::memcpy(&serialized_tensor[0], &v[0], num_elements * sizeof(T));
  return serialized_tensor;
}

template<typename T>
inline metisfl::TensorQuantifier QuantifyTensor(const metisfl::TensorSpec &tensor_spec) {
  /*
   * This function returns the tensor measurements. The first item in the returned
   * tuple represents the number of non-zero elements, the second item the number of
   * zero elements and the last item the size of the tensor in bytes.
   */
  auto t = DeserializeTensor<T>(tensor_spec);
  auto t_zeros = std::count(t.begin(), t.end(), 0);
  auto t_non_zeros = t.size() - t_zeros;
  auto t_bytes = sizeof(T) * t.size();
  auto tensor_quantifier = metisfl::TensorQuantifier();
  tensor_quantifier.set_tensor_non_zeros(t_non_zeros);
  tensor_quantifier.set_tensor_zeros(t_zeros);
  tensor_quantifier.set_tensor_size_bytes(t_bytes);
  return tensor_quantifier;
}

std::vector<char> GenSerializedEmptyTensor(const metisfl::TensorSpec &tensor_spec) {

  /**
   * Creates a tensor of the given size and data type with zero values.
   * The serializes the tensor and converts its serialized version to string.
   */
  std::vector<char> serialized_tensor;
  auto num_values = tensor_spec.length();
  auto data_type = tensor_spec.type().type();

  if (data_type == metisfl::DType_Type_UINT8) {
    serialized_tensor = SerializeTensor<unsigned char>(std::vector<unsigned char>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT16) {
    serialized_tensor = SerializeTensor<unsigned short>(std::vector<unsigned short>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT32) {
    serialized_tensor = SerializeTensor<unsigned int>(std::vector<unsigned int>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT64) {
    serialized_tensor = SerializeTensor<unsigned long>(std::vector<unsigned long>(num_values));
  } else if (data_type == metisfl::DType_Type_INT8) {
    serialized_tensor = SerializeTensor<signed char>(std::vector<signed char>(num_values));
  } else if (data_type == metisfl::DType_Type_INT16) {
    serialized_tensor = SerializeTensor<signed short>(std::vector<signed short>(num_values));
  } else if (data_type == metisfl::DType_Type_INT32) {
    serialized_tensor = SerializeTensor<signed int>(std::vector<signed int>(num_values));
  } else if (data_type == metisfl::DType_Type_INT64) {
    serialized_tensor = SerializeTensor<signed long>(std::vector<signed long>(num_values));
  } else if (data_type == metisfl::DType_Type_FLOAT32) {
    serialized_tensor = SerializeTensor<float>(std::vector<float>(num_values));
  } else if (data_type == metisfl::DType_Type_FLOAT64) {
    serialized_tensor = SerializeTensor<double>(std::vector<double>(num_values));
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }
  return serialized_tensor;

}

std::vector<char> GenEmptyTensor(const metisfl::TensorSpec &tensor_spec) {

  /**
   * Creates a tensor of the given size and data type with zero values.
   * The serializes the tensor and converts its serialized version to string.
   */
  std::vector<char> serialized_tensor;
  auto num_values = tensor_spec.length();
  auto data_type = tensor_spec.type().type();

  if (data_type == metisfl::DType_Type_UINT8) {
    serialized_tensor = SerializeTensor<unsigned char>(std::vector<unsigned char>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT16) {
    serialized_tensor = SerializeTensor<unsigned short>(std::vector<unsigned short>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT32) {
    serialized_tensor = SerializeTensor<unsigned int>(std::vector<unsigned int>(num_values));
  } else if (data_type == metisfl::DType_Type_UINT64) {
    serialized_tensor = SerializeTensor<unsigned long>(std::vector<unsigned long>(num_values));
  } else if (data_type == metisfl::DType_Type_INT8) {
    serialized_tensor = SerializeTensor<signed char>(std::vector<signed char>(num_values));
  } else if (data_type == metisfl::DType_Type_INT16) {
    serialized_tensor = SerializeTensor<signed short>(std::vector<signed short>(num_values));
  } else if (data_type == metisfl::DType_Type_INT32) {
    serialized_tensor = SerializeTensor<signed int>(std::vector<signed int>(num_values));
  } else if (data_type == metisfl::DType_Type_INT64) {
    serialized_tensor = SerializeTensor<signed long>(std::vector<signed long>(num_values));
  } else if (data_type == metisfl::DType_Type_FLOAT32) {
    serialized_tensor = SerializeTensor<float>(std::vector<float>(num_values));
  } else if (data_type == metisfl::DType_Type_FLOAT64) {
    serialized_tensor = SerializeTensor<double>(std::vector<double>(num_values));
  } else {
    throw std::runtime_error("Unsupported tensor data type.");
  }
  return serialized_tensor;

}

template<typename T>
inline void PrintSerializedTensor(const std::string &str, const uint32_t num_values) {
  std::vector<T> loaded_values(num_values);
  std::memcpy(&loaded_values[0], str.c_str(), num_values * sizeof(T));
  for (auto val: loaded_values) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;
}

} // namespace
} // namespace proto

#endif //METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
