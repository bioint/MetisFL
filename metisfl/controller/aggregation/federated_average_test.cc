
#include "metisfl/controller/aggregation/federated_average.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

using metisfl::proto::TensorOps;
using ::testing::proto::EqualsProto;

const char kModel1_with_tensor_values_1to10_as_FLOAT32[] = R"pb(
  encrypted: false
  tensors {
    length: 10
    dimensions: 10
    value: "\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A\000\000\020A\000\000 A"
  }
)pb";

const char kModel1_with_tensor_values_1to10_as_FLOAT64[] = R"pb(
  encrypted: false
  tensors {
    length: 10
    dimensions: 10
    value: "\000\000\000\000\000\000\360?\000\000\000\000\000\000\000@\000\000\000\000\000\000\010@\000\000\000\000\000\000\020@\000\000\000\000\000\000\024@\000\000\000\000\000\000\030@\000\000\000\000\000\000\034@\000\000\000\000\000\000 @\000\000\000\000\000\000\"@\000\000\000\000\000\000$@"
  }
)pb";

class FederatedAverageTest : public ::testing::Test {};

TEST_F(FederatedAverageTest, CorrectAverageFLOAT32) {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT32);

  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  FederatedAverage avg = FederatedAverage();
  Model averaged = avg.Aggregate(to_aggregate);

  auto aggregated_value_serialized = averaged.tensors().at(0).value();
  auto num_values = averaged.tensors().at(0).length();
  TensorOps::PrintSerializedTensor(aggregated_value_serialized, num_values);
  EXPECT_THAT(averaged, EqualsProto(model1));
}

TEST_F(FederatedAverageTest, CorrectAverageFLOAT64) {
  auto model1 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT64);
  auto model2 = TensorOps::ParseTextOrDie<Model>(
      kModel1_with_tensor_values_1to10_as_FLOAT64);

  std::vector seq1({std::make_pair<const Model *, double>(&model1, 0.5)});
  std::vector seq2({std::make_pair<const Model *, double>(&model2, 0.5)});
  std::vector to_aggregate({seq1, seq2});

  FederatedAverage avg = FederatedAverage();
  Model averaged = avg.Aggregate(to_aggregate);
  auto aggregated_value_serialized = averaged.tensors().at(0).value();
  auto num_values = averaged.tensors().at(0).length();
  TensorOps::PrintSerializedTensor(aggregated_value_serialized, num_values);

  // The aggregated value should be equal with the half of models sum.
  // Therefore, the global model must be equal to either model1 or model2.
  EXPECT_THAT(averaged, EqualsProto(model1));
}

}  // namespace metisfl::controller