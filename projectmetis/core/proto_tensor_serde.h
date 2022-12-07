
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CORE_PROTO_TENSOR_SERDE_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CORE_PROTO_TENSOR_SERDE_H_

#include "projectmetis/proto/model.pb.h"

#include <cstring>
#include <vector>

namespace proto {

namespace {
template<typename T>
inline std::vector<T> DeserializeTensor(const projectmetis::TensorSpec &tensor_spec) {
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
inline projectmetis::TensorQuantifier QuantifyTensor(const projectmetis::TensorSpec &tensor_spec) {
  /*
   * This function returns the tensor measurements. The first item in the returned
   * tuple represents the number of non-zero elements, the second item the number of
   * zero elements and the last item the size of the tensor in bytes.
   */
  auto t = DeserializeTensor<T>(tensor_spec);
  auto t_zeros = std::count(t.begin(), t.end(), 0);
  auto t_non_zeros = t.size() - t_zeros;
  auto t_bytes = sizeof(T) * t.size();
  auto tensor_quantifier = projectmetis::TensorQuantifier();
  tensor_quantifier.set_tensor_non_zeros(t_non_zeros);
  tensor_quantifier.set_tensor_zeros(t_zeros);
  tensor_quantifier.set_tensor_size_bytes(t_bytes);
  return tensor_quantifier;
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

#endif // PROJECTMETIS_RC_PROJECTMETIS_CORE_PROTO_TENSOR_SERDE_H_
