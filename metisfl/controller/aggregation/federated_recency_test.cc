
#include "metisfl/controller/aggregation/federated_recency.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "metisfl/controller/common/macros.h"
#include "metisfl/controller/common/proto_matchers.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/proto/model.pb.h"

using metisfl::proto::DeserializeTensor;
using metisfl::proto::ParseTextOrDie;
using metisfl::proto::PrintSerializedTensor;
using metisfl::proto::SerializeTensor;
using ::testing::proto::EqualsProto;
using namespace std;

namespace metisfl::controller {
namespace {

const char kModel1_with_tensor_values_1to10_as_INT32[] = R"pb(
  tensor {
    length: 10
    encrypted: false
    dimensions: 10
    type { type: INT32 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
    value: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\006\000\000\000\007\000\000\000\010\000\000\000\t\000\000\000\n\000\000\000"
  }
)pb";

const char kModel1_with_tensor_values_1to10_as_FLOAT32[] = R"pb(
  tensor {
    length: 10
    encrypted: false
    dimensions: 10
    type { type: FLOAT32 byte_order: LITTLE_ENDIAN_ORDER fortran_order: False }
    value: "\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A\000\000\020A\000\000 A"
  }
)pb";

class FederatedRecencyTest : public ::testing::Test {
 public:
  static Model RecencyAggregation(
      const std::vector<std::vector<std::pair<Model *, double>>> &to_aggregate) {
    auto aggregator_ = FederatedRecency<double>();
    Model averaged_;
    for (auto &itr : to_aggregate) {
      // Wrap learner's models within a new vector to be sent for aggregation.
      std::vector<std::vector<std::pair<Model *, double>>> tmp{itr};
      averaged_ = aggregator_.Aggregate(tmp);
    }
    return averaged_;
  }
};

TEST_F(FederatedRecencyTest, ValidateLearnerLineageLength) /* NOLINT */ {
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_INT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1),
                    std::make_pair<const Model *, double>(&model1, 1),
                    std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);

  // The returned Federated Model proto message has no model proto message.
  EXPECT_FALSE(averaged.has_model());
}

TEST_F(FederatedRecencyTest, ModelFloat32SingleLearnerOneModel) /* NOLINT */ {
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_INT32);
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1)});
  std::vector to_aggregate({seq1});

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);
  auto averaged_value_serialized = averaged.model().tensors().at(0).value();
  auto num_values = averaged.model().tensors().at(0).length();
  metisfl::proto::PrintSerializedTensor<float>(averaged_value_serialized,
                                               num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged.model(), EqualsProto(model1));
}

TEST_F(FederatedRecencyTest, ModelFLoat32SingleLearnerTwoModels) /* NOLINT */ {
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  auto model2 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Modify second's model variable value by multiplying by 2.
  auto variable = DeserializeTensor<float>(model1.tensors().at(0));
  for (auto &val : variable) val *= 2;
  auto variable_ser = SerializeTensor<float>(variable);
  std::string variable_ser_str(variable_ser.begin(), variable_ser.end());
  *model2.mutable_tensors(0)->mutable_value() = variable_ser_str;

  // Create the learner models pair.
  std::vector seq1({std::make_pair<const Model *, double>(&model1, 1),
                    std::make_pair<const Model *, double>(&model2, 1)});
  std::vector to_aggregate({seq1});

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);
  auto averaged_value_serialized = averaged.model().tensors().at(0).value();
  auto num_values = averaged.model().tensors().at(0).length();
  metisfl::proto::PrintSerializedTensor<float>(averaged_value_serialized,
                                               num_values);

  // We expect that the returned federated/averaged model to be equal
  // to the second model that was part of the learner's model collection.
  EXPECT_THAT(averaged.model(), EqualsProto(model2));
}

TEST_F(FederatedRecencyTest, ModelFloat32AggregationFirstTimeCommittersV1)
/* NOLINT */ {
  // Model for the first time committing / requesting aggregation.
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the second time committing / requesting aggregation.
  auto model2 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the third time committing / requesting aggregation.
  auto model3 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Construct model sequence for each learner. The main
  // difference across all learners is the contribution value.
  // Learner 1.
  std::vector learner1_first_time(
      {std::make_pair<const Model *, double>(&model1, 1)});
  // Learner 2.
  std::vector learner2_first_time(
      {std::make_pair<const Model *, double>(&model2, 2)});
  // Learner 3.
  std::vector learner3_first_time(
      {std::make_pair<const Model *, double>(&model3, 3)});

  // Construct model aggregation sequence from all learners.
  std::vector to_aggregate({
      learner1_first_time,
      learner2_first_time,
      learner3_first_time,
  });

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);
  auto averaged_value_serialized = averaged.model().tensors().at(0).value();
  auto num_values = averaged.model().tensors().at(0).length();
  metisfl::proto::PrintSerializedTensor<float>(averaged_value_serialized,
                                               num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged.model(), EqualsProto(model1));
}

TEST_F(FederatedRecencyTest, ModelFloat32AggregationFirstTimeCommittersV2)
/* NOLINT */ {
  // Model for the first time committing / requesting aggregation.
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the second time committing / requesting aggregation.
  auto model2 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the third time committing / requesting aggregation.
  auto model3 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Modify third learner's model variable value by multiplying by 0.
  auto variable = DeserializeTensor<float>(model3.tensors().at(0));
  for (auto &val : variable) val *= 0;
  auto variable_ser = SerializeTensor<float>(variable);
  std::string variable_ser_str(variable_ser.begin(), variable_ser.end());
  *model3.mutable_tensors(0)->mutable_value() = variable_ser_str;

  // Construct model sequence for each learner. The main
  // difference across all learners is the contribution value.
  // Learner 1.
  std::vector learner1_first_time(
      {std::make_pair<const Model *, double>(&model1, 1)});
  // Learner 2.
  std::vector learner2_first_time(
      {std::make_pair<const Model *, double>(&model2, 2)});
  // Learner 3.
  std::vector learner3_first_time(
      {std::make_pair<const Model *, double>(&model3, 3)});

  // Construct model aggregation sequence from all learners.
  std::vector to_aggregate({
      learner1_first_time,
      learner2_first_time,
      learner3_first_time,
  });

  // Expected outcome:
  // Models Normalization Factor: (1 + 2 + 3) = 6
  // Weighting factor for learner1's model: 1 / 6 = 0.16
  // Weighting factor for learner2's model: 2 / 6 = 0.33
  // Weighting factor for learner3's model: 3 / 6 = 0.5
  // Averaged Model: ( 0.16 * 1 + 0.33 * 1 + 0.5 * 0
  //                    | 0.16 * 2 + 0.33 * 2 + 0.5 * 0
  //                      | 0.16 * 3 + 0.33 * 3 + 0.5 * 0 | ... )
  //                 ( 0.49 | 0.98 | 1.47 | 1.96 | 2.45 | 2.96 | 3.43 | 3.92
  //                 | 4.41 | 4.9 )
  // Should Be       (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 )
  auto expected =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector<float> expected_values{0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5};
  auto serialized_tensor = SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);
  auto averaged_value_serialized = averaged.model().tensors().at(0).value();
  auto num_values = averaged.model().tensors().at(0).length();
  metisfl::proto::PrintSerializedTensor<float>(averaged_value_serialized,
                                               num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged.model(), EqualsProto(expected));
}

TEST_F(FederatedRecencyTest, ModelFloat32AggregationOneTimeCommitters)
/* NOLINT */ {
  // Model for the first time committing / requesting aggregation.
  auto model1 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the second time committing / requesting aggregation.
  auto model2 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  // Model for the third time committing / requesting aggregation.
  auto model3 =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);

  // Modify second's model variable value by multiplying by 2.
  auto variable = DeserializeTensor<float>(model3.tensors().at(0));
  for (auto &val : variable) val *= 0.5;
  auto variable_ser = SerializeTensor<float>(variable);
  std::string variable_ser_str(variable_ser.begin(), variable_ser.end());
  *model3.mutable_tensors(0)->mutable_value() = variable_ser_str;

  // Construct model sequence for each learner. The main
  // difference across all learners is the contribution value.
  // Learner 1.
  std::vector learner1_first_time(
      {std::make_pair<const Model *, double>(&model1, 1)});
  std::vector learner1_second_time(
      {std::make_pair<const Model *, double>(&model1, 1),
       std::make_pair<const Model *, double>(&model2, 1)});
  std::vector learner1_third_time(
      {std::make_pair<const Model *, double>(&model2, 1),
       std::make_pair<const Model *, double>(&model3, 1)});
  // Learner 2.
  std::vector learner2_first_time(
      {std::make_pair<const Model *, double>(&model1, 2)});
  std::vector learner2_second_time(
      {std::make_pair<const Model *, double>(&model1, 2),
       std::make_pair<const Model *, double>(&model2, 2)});
  std::vector learner2_third_time(
      {std::make_pair<const Model *, double>(&model2, 2),
       std::make_pair<const Model *, double>(&model3, 2)});
  // Learner 3.
  std::vector learner3_first_time(
      {std::make_pair<const Model *, double>(&model1, 3)});

  // Construct model aggregation sequence from all learners.
  std::vector to_aggregate({learner1_first_time, learner2_first_time,
                            learner1_second_time, learner2_second_time,
                            learner3_first_time, learner1_third_time,
                            learner2_third_time});

  /*
   Expected Outcome
   Models Normalization Factor: 6
   Final Scaled Model:{4.5, 9, 13.5, 18, 22.5, 27, 31.5, 36, 40.5, 45}
   Final Community Model:{4.5, 9, 13.5, 18, 22.5, 27, 31.5, 36, 40.5, 45}*1/6
   Final Community Model:{0.75, 1.5, 2.25, 3, 3.75, 4.5, 5.25, 6, 6.75, 7.5}
  */
  auto expected =
      ParseTextOrDie<Model>(kModel1_with_tensor_values_1to10_as_FLOAT32);
  std::vector<float> expected_values{0.75, 1.5,  2.25, 3,    3.75,
                                     4.5,  5.25, 6,    6.75, 7.5};
  auto serialized_tensor = SerializeTensor(expected_values);
  std::string serialized_tensor_str(serialized_tensor.begin(),
                                    serialized_tensor.end());
  *expected.mutable_tensors(0)->mutable_value() = serialized_tensor_str;

  auto averaged = FederatedRecencyTest::RecencyAggregation(to_aggregate);
  auto averaged_value_serialized = averaged.model().tensors().at(0).value();
  auto num_values = averaged.model().tensors().at(0).length();
  metisfl::proto::PrintSerializedTensor<float>(averaged_value_serialized,
                                               num_values);

  // The aggregated model proto definition should be equal to the original
  // model.
  EXPECT_THAT(averaged.model(), EqualsProto(expected));
}

}  // namespace
}  // namespace metisfl::controller
