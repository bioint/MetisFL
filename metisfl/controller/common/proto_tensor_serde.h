
#ifndef METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
#define METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_

#include <cstring>
#include <vector>

#include "metisfl/proto/model.pb.h"

namespace metisfl {
namespace proto {
template <typename T>
class ProtoSerde {
 public:
  static std::vector<T> DeserializeTensor(const metisfl::Tensor &tensor);

  static std::vector<char> SerializeTensor(const std::vector<T> &v);

  static metisfl::TensorQuantifier QuantifyTensor(
      const metisfl::Tensor &tensor);

  static void PrintSerializedTensor(const std::string &str,
                                    const uint32_t num_values);
};

}  // namespace proto

}  // namespace metisfl

#endif  // METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
