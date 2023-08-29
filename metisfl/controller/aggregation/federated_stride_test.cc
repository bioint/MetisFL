
#include "metisfl/controller/aggregation/federated_stride.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {
namespace {

using metisfl::proto::TensorOps;
using ::testing::proto::EqualsProto;
using namespace std;

const char kModel1_with_tensor_values_1to10_as_INT32[] = R"pb(
  encrypted: false
  tensor {
    length: 10
    dimensions: 10
    value: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\006\000\000\000\007\000\000\000\010\000\000\000\t\000\000\000\n\000\000\000"
  }
)pb";

const char kModel1_with_tensor_values_1to10_as_FLOAT32[] = R"pb(
  encrypted: false
  tensors {
    length: 10
    dimensions: 10
    value: "\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A\000\000\020A\000\000 A"
  }
)pb";

class FederatedStrideTest : public ::testing::Test {
 public:
  static Model StridedAggregation(
      const std::vector<std::vector<std::pair<const Model *, double>>>
          &to_aggregate,
      const uint32_t &stride_length) {
    // Construct batch of stride_length and perform aggregation.
    Model averaged;
    auto aggregator = FederatedStride();
    std::vector<std::vector<std::pair<const Model *, double>>> to_stride;
    for (auto itr = to_aggregate.begin(); itr != to_aggregate.end(); itr++) {
      to_stride.push_back({*itr});
      if (to_stride.size() == stride_length || itr == (--to_aggregate.end())) {
        averaged = aggregator.Aggregate(to_stride);
        to_stride.clear();
      }
    }
    aggregator.Reset();
    return averaged;
  }
};

TEST_F(FederatedStrideTest, ModelInt32SingleLearnerStride1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  auto expected = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(expected));
}

TEST_F(FederatedStrideTest, ModelInt32TwoLearnersStride1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Stride: 1
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  auto expected = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  // CAUTION! Since the expected values are of type int32 there is no precision.
  // Therefore, similar to uint16, everything is rounded down to the closest
  // integer:
  //    uint32(0.5 * 1) + uint32(0.5 * 1) = uint32(0.5) + uint32(0.5) = 0
  //    uint32(0.5 * 9) + uint32(0.5 * 9) = uint32(4.5) + uint32(4.5) = 8
  std::vector expected_values{0, 2, 2, 4, 4, 6, 6, 8, 8, 10};
  auto serialized_tensor = TensorOps::SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(expected));
}

TEST_F(FederatedStrideTest, ModelInt32FourLearnersStride1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  auto model3 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  auto model4 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.25)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.25)});
  std::vector seq3({std::make_pair<const Model *, double>(&model3, 0.25)});
  std::vector seq4({std::make_pair<const Model *, double>(&model4, 0.25)});
  std::vector to_aggregate({seq1, seq2, seq3, seq4});

  // Stride: 1
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  auto expected = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_INT32);
  // CAUTION! Since the expected values are of type int32 there is no precision.
  // Therefore, similar to uint16, everything is rounded down to the closest
  // integer:
  //    uint32(0.5 * 1) + uint32(0.5 * 1) = uint32(0.5) + uint32(0.5) = 0
  //    uint32(0.5 * 9) + uint32(0.5 * 9) = uint32(4.5) + uint32(4.5) = 8
  std::vector expected_values{0, 0, 0, 4, 4, 4, 4, 8, 8, 8};
  auto serialized_tensor = TensorOps::SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(expected));
}

TEST_F(FederatedStrideTest, ModelFloat32SingleLearnerStride1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  // Stride: 1
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32TwoLearnersStride1V1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Stride: 1
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32TwoLearnersStride1V2) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Modify second model's variable value by multiplying by 2.
  auto variable = TensorOps::DeserializeTensor(model2.tensors().at(0));
  for (auto &val : variable) val *= 2;
  auto variable_ser = TensorOps::SerializeTensor(variable);
  std::string variable_ser_str(variable_ser.begin(), variable_ser.end());
  *model2.mutable_tensors(0)->mutable_value() = variable_ser_str;

  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // Stride: 1
  // Expected outcome:
  // Models Normalization Factor: (0.5 + 0.5) = 1
  // Weighting factor for learner1's model: 0.5 / 1 = 0.5
  // Weighting factor for learner2's model: 0.5 / 1 = 0.5
  // Averaged Model: ( 0.5 * 1 + 0.5 * 2 | 0.5 * 2 + 0.5 * 4 | 0.5 * 3 + 0.5 * 8
  // | ... )
  //                 ( 1.5 | 3 | 4.5 | 6 | 7.5 | 9 | 10.5 | 12 | 13.5 | 15 )
  auto expected = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector expected_values{1.5, 3.0,  4.5,  6.0,  7.5,
                              9.0, 10.5, 12.0, 13.5, 15.0};
  auto serialized_tensor = TensorOps::SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(expected));
}

TEST_F(FederatedStrideTest, ModelFloat32FourLearnersStride1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model3 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model4 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.25)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.25)});
  std::vector seq3({std::make_pair<const Model *, double>(&model3, 0.25)});
  std::vector seq4({std::make_pair<const Model *, double>(&model4, 0.25)});
  std::vector to_aggregate({seq1, seq2, seq3, seq4});

  // Stride: 1
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32SingleLearnerStride2) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  // Stride: 2
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 1);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32TwoLearnersStride2) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Stride: 2
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 2);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32FourLearnersStride2) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model3 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model4 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.25)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.25)});
  std::vector seq3({std::make_pair<const Model *, double>(&model3, 0.25)});
  std::vector seq4({std::make_pair<const Model *, double>(&model4, 0.25)});
  std::vector to_aggregate({seq1, seq2, seq3, seq4});

  // Stride: 2
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 2);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32SingleLearnerStride3) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  // Stride: 3
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 3);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32TwoLearnersStride3) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  // Stride: 3
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 3);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32FourLearnersStride3V1) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model3 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model4 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.25)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.25)});
  std::vector seq3({std::make_pair<const Model *, double>(&model3, 0.25)});
  std::vector seq4({std::make_pair<const Model *, double>(&model4, 0.25)});
  std::vector to_aggregate({seq1, seq2, seq3, seq4});

  // Stride: 3
  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 3);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedStrideTest, ModelFloat32FourLearnersStride3V2) /* NOLINT */ {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model3 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model4 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Modify fourth model's variable value by multiplying by 0.
  auto variable = TensorOps::DeserializeTensor(model4.tensors().at(0));
  for (auto &val : variable) val *= 0;
  auto variable_ser = TensorOps::SerializeTensor(variable);
  std::string variable_ser_str(variable_ser.begin(), variable_ser.end());
  *model4.mutable_tensors(0)->mutable_value() = variable_ser_str;

  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.25)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.25)});
  std::vector seq3({std::make_pair<const Model *, double>(&model3, 0.25)});
  std::vector seq4({std::make_pair<const Model *, double>(&model4, 0.25)});
  std::vector to_aggregate({seq1, seq2, seq3, seq4});

  // Stride: 3
  // Expected outcome:
  // Models Normalization Factor: (0.25 + 0.25 + 0.25 + 0.25) = 1
  // Weighting factor for learner1's model: 0.25 / 1 = 0.25
  // Weighting factor for learner2's model: 0.25 / 1 = 0.25
  // Weighting factor for learner3's model: 0.25 / 1 = 0.25
  // Weighting factor for learner4's model: 0.5 / 1 = 0.25
  // Averaged Model: ( 0.25 * 1 + 0.25 * 1 + 0.25 * 1 + 0.25 * 0
  //                      | 0.25 * 2 + 0.25 * 2 + 0.25 * 2 + 0.25 * 0
  //                        | 0.25 * 3 + 0.25 * 3 + 0.25 * 3 + 0.25 * 0
  //                          | ...
  //                 ( 0.75 | 1.5 | 2.25 | 3 | 3.75 | 4.5 | 5.25 | 6 | 6.75
  //                 | 7.5)
  auto expected = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector expected_values{0.75, 1.5,  2.25, 3,    3.75,
                              4.5,  5.25, 6,    6.75, 7.5};
  auto serialized_tensor = TensorOps::SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  auto averaged = FederatedStrideTest::StridedAggregation(to_aggregate, 3);
  auto averaged_value_serialized = averaged.tensors(0).value();
  auto num_values = averaged.tensors(0).length();
  PrintSerializedTensor(averaged_value_serialized, num_values);

  // The aggregated model proto definition should be equal to any of the
  // original models.
  EXPECT_THAT(averaged, EqualsProto(expected));
}

}  // namespace
}  // namespace metisfl::controller
