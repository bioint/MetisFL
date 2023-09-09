
#ifndef METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
#define METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_

#include <google/protobuf/text_format.h>

#include <cstring>
#include <vector>

#include "metisfl/controller/common/macros.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl {
namespace proto {

class TensorOps {
 public:
  static std::vector<double> DeserializeTensor(const metisfl::Tensor &tensor) {
    const auto tensor_bytes = tensor.value().c_str();
    const auto tensor_elements_num = tensor.length();
    std::vector<double> deserialized_tensor(tensor_elements_num);
    std::memcpy(&deserialized_tensor[0], tensor_bytes,
                tensor_elements_num * sizeof(double));
    return deserialized_tensor;
  }

  static std::vector<char> SerializeTensor(const std::vector<double> &v) {
    auto num_elements = v.size();
    std::vector<char> serialized_tensor(num_elements * sizeof(double));
    std::memcpy(&serialized_tensor[0], &v[0], num_elements * sizeof(double));
    return serialized_tensor;
  }

  static void PrintSerializedTensor(const std::string &str,
                                    const uint32_t num_values) {
    std::vector<double> loaded_values(num_values);
    std::memcpy(&loaded_values[0], str.c_str(), num_values * sizeof(double));
    for (auto val : loaded_values) {
      std::cout << val << ", ";
    }
    std::cout << std::endl;
  }

  template <typename T>
  static T ParseTextOrDie(const std::string &input) {
    T result;
    VALIDATE(google::protobuf::TextFormat::ParseFromString(input, &result));
    return result;
  }
};
}  // namespace proto

}  // namespace metisfl

#endif  // METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_