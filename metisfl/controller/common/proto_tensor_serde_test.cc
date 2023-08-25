
#include "metisfl/controller/common/proto_tensor_serde.h"

#include <gtest/gtest.h>

#include <iostream>

#include "metisfl/controller/common/macros.h"
#include "metisfl/proto/model.pb.h"

namespace proto {
namespace {
using metisfl::proto::DeserializeTensor;
using metisfl::proto::SerializeTensor;
using metisfl::proto::ParseTextOrDie;;

// The value below shows the byte representation of unsigned short [1 to 10].
char kTensor_1to10_as_UINT16[] = R"pb(
  length: 10
  dimensions: 10
  type { type: UINT16 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
  value: "\001\000\002\000\003\000\004\000\005\000\006\000\007\000\010\000\t\000\n\000"
)pb";

// The value below shows the byte representation of signed integer [1 to 10].
char kTensor_1to10_as_INT32[] = R"pb(
  length: 10
  dimensions: 10
  type { type: INT32 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
  value: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\006\000\000\000\007\000\000\000\010\000\000\000\t\000\000\000\n\000\000\000"
)pb";

// The value below shows the byte representation of floats [1.0 to 10.0].
char kTensor_1to10_as_FLOAT32[] = R"pb(
  length: 10
  dimensions: 10
  type { type: FLOAT32 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
  value: "\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A\000\000\020A\000\000 A"
)pb";

// The value below shows the byte representation of doubles [1.0 to 10.0].
char kTensor_1to10_as_FLOAT64[] = R"pb(
  length: 10
  dimensions: 10
  type { type: FLOAT64 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
  value: "\000\000\000\000\000\000\360?\000\000\000\000\000\000\000@\000\000\000\000\000\000\010@\000\000\000\000\000\000\020@\000\000\000\000\000\000\024@\000\000\000\000\000\000\030@\000\000\000\000\000\000\034@\000\000\000\000\000\000 @\000\000\000\000\000\000\"@\000\000\000\000\000\000$@"
)pb";

class ProtoTensorSerDe : public ::testing::Test {};

TEST_F(ProtoTensorSerDe, DeSerUINT16) /* NOLINT */ {
  auto tensor_uint16 = ParseTextOrDie<metisfl::Tensor>(kTensor_1to10_as_UINT16);
  auto num_values = tensor_uint16.length();
  auto deserialized_tensor = DeserializeTensor<unsigned short>(tensor_uint16);
  auto serialized_tensor = SerializeTensor<unsigned short>(deserialized_tensor);

  // Deserialize the serialized values to validate equality with original
  // tensor.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  std::vector<unsigned short> deserialized_tensor_aux(num_values);
  std::memcpy(&deserialized_tensor_aux[0], serialized_tensor_str.c_str(),
              num_values * sizeof(unsigned short));

  for (auto val : deserialized_tensor_aux) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;

  // Validate vectors equality using the (==) operator overload.
  bool are_vectors_equal = deserialized_tensor == deserialized_tensor_aux;
  EXPECT_TRUE(are_vectors_equal);
}

TEST_F(ProtoTensorSerDe, DeSerINT32) /* NOLINT */ {
  auto tensor_int32 = ParseTextOrDie<metisfl::Tensor>(kTensor_1to10_as_INT32);
  auto num_values = tensor_int32.length();
  auto deserialized_tensor = DeserializeTensor<signed int>(tensor_int32);
  auto serialized_tensor = SerializeTensor<signed int>(deserialized_tensor);

  // Deserialize the serialized values to validate equality with original
  // tensor.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  std::vector<signed int> deserialized_tensor_aux(num_values);
  std::memcpy(&deserialized_tensor_aux[0], serialized_tensor_str.c_str(),
              num_values * sizeof(signed int));

  for (auto val : deserialized_tensor_aux) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;

  // Validate vectors equality using the (==) operator overload.
  bool are_vectors_equal = deserialized_tensor == deserialized_tensor_aux;
  EXPECT_TRUE(are_vectors_equal);
}

TEST_F(ProtoTensorSerDe, DeSerFLOAT32) /* NOLINT */ {
  auto tensor_float32 =
      ParseTextOrDie<metisfl::Tensor>(kTensor_1to10_as_FLOAT32);
  auto num_values = tensor_float32.length();
  auto deserialized_tensor = DeserializeTensor<float>(tensor_float32);
  auto serialized_tensor = SerializeTensor<float>(deserialized_tensor);

  // Deserialize the serialized values to validate equality with original
  // tensor.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  std::vector<float> deserialized_tensor_aux(num_values);
  std::memcpy(&deserialized_tensor_aux[0], serialized_tensor_str.c_str(),
              num_values * sizeof(float));

  for (auto val : deserialized_tensor_aux) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;

  // Validate vectors equality using the (==) operator overload.
  bool are_vectors_equal = deserialized_tensor == deserialized_tensor_aux;
  EXPECT_TRUE(are_vectors_equal);
}

TEST_F(ProtoTensorSerDe, DeSerFLOAT64) /* NOLINT */ {
  auto tensor_float64 =
      ParseTextOrDie<metisfl::Tensor>(kTensor_1to10_as_FLOAT64);
  auto num_values = tensor_float64.length();
  auto deserialized_tensor = DeserializeTensor<double>(tensor_float64);
  auto serialized_tensor = SerializeTensor<double>(deserialized_tensor);

  // Deserialize the serialized values to validate equality with original
  // tensor.
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  std::vector<double> deserialized_tensor_aux(num_values);
  std::memcpy(&deserialized_tensor_aux[0], serialized_tensor_str.c_str(),
              num_values * sizeof(double));

  for (auto val : deserialized_tensor_aux) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;

  // Validate vectors equality using the (==) operator overload.
  bool are_vectors_equal = deserialized_tensor == deserialized_tensor_aux;
  EXPECT_TRUE(are_vectors_equal);
}

}  // namespace
}  // namespace proto
