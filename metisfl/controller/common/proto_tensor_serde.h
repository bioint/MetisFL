
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
  std::vector<T> DeserializeTensor(const metisfl::Tensor &tensor);

  std::vector<char> SerializeTensor(const std::vector<T> &v);

  metisfl::TensorQuantifier QuantifyTensor(const metisfl::Tensor &tensor);

  std::vector<T> GenSerializedEmptyTensor(const metisfl::Tensor &tensor);

  void PrintSerializedTensor(const std::string &str, const uint32_t num_values);

  static ProtoSerde<T> GetProtoSerde(DType_Type dtype);
};

}  // namespace proto

}  // namespace metisfl

#endif  // METISFL_METISFL_CONTROLLER_COMMON_PROTO_TENSOR_SERDE_H_
