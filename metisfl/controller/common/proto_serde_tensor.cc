
#include "metisfl/controller/common/proto_tensor_serde.h"

namespace metisfl {
namespace proto {
template <typename T>
class ProtoSerde {
 public:
  std::vector<T> DeserializeTensor(const metisfl::Tensor &tensor) {
    const auto tensor_bytes = tensor.value().c_str();
    const auto tensor_elements_num = tensor.length();
    std::vector<T> deserialized_tensor(tensor_elements_num);
    std::memcpy(&deserialized_tensor[0], tensor_bytes,
                tensor_elements_num * sizeof(T));
    return deserialized_tensor;
  }

  std::vector<char> SerializeTensor(const std::vector<T> &v) {
    auto num_elements = v.size();
    std::vector<char> serialized_tensor(num_elements * sizeof(T));
    std::memcpy(&serialized_tensor[0], &v[0], num_elements * sizeof(T));
    return serialized_tensor;
  }

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

  std::vector<T> GenSerializedEmptyTensor(const int num_elements) {
    return SerializeTensor<T>(std::vector<T>(num_elements));
  }

  std::vector<char> GenEmptyTensor(const metisfl::Tensor &tensor);

  void PrintSerializedTensor(const std::string &str,
                             const uint32_t num_values) {
    std::memcpy(&loaded_values[0], str.c_str(), num_values * sizeof(T));
    for (auto val : loaded_values) {
      std::cout << val << ", ";
    }
    std::cout << std::endl;
  }
  static ProtoSerde<T> GetProtoSerde(DType_Type dtype) {
    switch (dtype) {
      case DType_Type_UINT8:
        return ProtoSerde<unsigned char>();
      case DType_Type_UINT16:
        return ProtoSerde<unsigned short>();
      case DType_Type_UINT32:
        return ProtoSerde<unsigned int>();
      case DType_Type_UINT64:
        return ProtoSerde<unsigned long>();
      case DType_Type_INT8:
        return ProtoSerde<signed char>();
      case DType_Type_INT16:
        return ProtoSerde<signed short>();
      case DType_Type_INT32:
        return ProtoSerde<signed int>();
      case DType_Type_INT64:
        return ProtoSerde<signed long>();
      case DType_Type_FLOAT32:
        return ProtoSerde<float>();
      case DType_Type_FLOAT64:
        return ProtoSerde<double>();
      default:
        PLOG(FATAL) << "Unsupported tensor data type.";
    }
  };

}  // namespace proto

}  // namespace metisfl