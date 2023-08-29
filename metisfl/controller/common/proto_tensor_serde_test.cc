
#include "metisfl/controller/common/proto_tensor_serde.h"

#include <gtest/gtest.h>

#include <iostream>

#include "metisfl/controller/common/macros.h"
#include "metisfl/proto/model.pb.h"

namespace proto {
namespace {
using metisfl::proto::TensorOps;

// The value below shows the byte representation of doubles [1.0 to 10.0].
char kTensor_1to10_as_FLOAT64[] = R"pb(
  length: 10
  dimensions: 10
  value: "\000\000\000\000\000\000\360?\000\000\000\000\000\000\000@\000\000\000\000\000\000\010@\000\000\000\000\000\000\020@\000\000\000\000\000\000\024@\000\000\000\000\000\000\030@\000\000\000\000\000\000\034@\000\000\000\000\000\000 @\000\000\000\000\000\000\"@\000\000\000\000\000\000$@"
)pb";

class ProtoTensorSerDe : public ::testing::Test {};

TEST_F(ProtoTensorSerDe, DeSerFLOAT64) /* NOLINT */ {
  auto tensor_float64 =
      TensorOps::ParseTextOrDie<metisfl::Tensor>(kTensor_1to10_as_FLOAT64);
  auto num_values = tensor_float64.length();
  auto deserialized_tensor = TensorOps::DeserializeTensor(tensor_float64);
  auto serialized_tensor = TensorOps::SerializeTensor(deserialized_tensor);

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
