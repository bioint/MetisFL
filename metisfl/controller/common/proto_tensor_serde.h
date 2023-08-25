
#ifndef METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
#define METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_

#include <google/protobuf/text_format.h>

#include <cstring>
#include <vector>

#include "metisfl/controller/common/macros.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl {
namespace proto {

template <typename T>
std::vector<T> DeserializeTensor(const metisfl::Tensor &tensor) {
  const auto tensor_bytes = tensor.value().c_str();
  const auto tensor_elements_num = tensor.length();
  std::vector<T> deserialized_tensor(tensor_elements_num);
  std::memcpy(&deserialized_tensor[0], tensor_bytes,
              tensor_elements_num * sizeof(T));
  return deserialized_tensor;
}

template <typename T>
std::vector<char> SerializeTensor(const std::vector<T> &v) {
  auto num_elements = v.size();
  std::vector<char> serialized_tensor(num_elements * sizeof(T));
  std::memcpy(&serialized_tensor[0], &v[0], num_elements * sizeof(T));
  return serialized_tensor;
}

template <typename T>
metisfl::TensorQuantifier QuantifyTensor(const metisfl::Tensor &tensor) {
  auto t = DeserializeTensor<T>(tensor);
  auto t_zeros = std::count(t.begin(), t.end(), 0);
  auto t_non_zeros = t.size() - t_zeros;
  auto t_bytes = sizeof(T) * t.size();
  auto tensor_quantifier = metisfl::TensorQuantifier();
  tensor_quantifier.set_tensor_non_zeros(t_non_zeros);
  tensor_quantifier.set_tensor_zeros(t_zeros);
  tensor_quantifier.set_tensor_size_bytes(t_bytes);
  return tensor_quantifier;
}

template <typename T>
void PrintSerializedTensor(const std::string &str, const uint32_t num_values) {
  std::vector<T> loaded_values(num_values);
  std::memcpy(&loaded_values[0], str.c_str(), num_values * sizeof(T));
  for (auto val : loaded_values) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
T ParseTextOrDie(const std::string &input) {
  T result;
  VALIDATE(google::protobuf::TextFormat::ParseFromString(input, &result));
  return result;
}

}  // namespace proto

}  // namespace metisfl

#endif  // METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_