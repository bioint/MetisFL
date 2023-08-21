
#include "metisfl/controller/common/proto_tensor_serde.h"

namespace metisfl {
namespace proto {
template <typename T>
class ProtoSerde {
 public:
  static std::vector<T> DeserializeTensor(const metisfl::Tensor &tensor) {
    const auto tensor_bytes = tensor.value().c_str();
    const auto tensor_elements_num = tensor.length();
    std::vector<T> deserialized_tensor(tensor_elements_num);
    std::memcpy(&deserialized_tensor[0], tensor_bytes,
                tensor_elements_num * sizeof(T));
    return deserialized_tensor;
  }

  static std::vector<char> SerializeTensor(const std::vector<T> &v) {
    auto num_elements = v.size();
    std::vector<char> serialized_tensor(num_elements * sizeof(T));
    std::memcpy(&serialized_tensor[0], &v[0], num_elements * sizeof(T));
    return serialized_tensor;
  }

  static metisfl::TensorQuantifier QuantifyTensor(
      const metisfl::Tensor &tensor) {
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

  static void PrintSerializedTensor(const std::string &str,
                                    const uint32_t num_values) {
    std::memcpy(&loaded_values[0], str.c_str(), num_values * sizeof(T));
    for (auto val : loaded_values) {
      std::cout << val << ", ";
    }
    std::cout << std::endl;
  }
};
}  // namespace proto

}  // namespace metisfl