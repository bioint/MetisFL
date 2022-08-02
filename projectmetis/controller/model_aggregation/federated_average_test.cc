
#include "projectmetis/controller/model_aggregation/federated_average.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "projectmetis/core/macros.h"
#include "projectmetis/proto/model.pb.h"
#include "projectmetis/core/matchers/proto_matchers.h"

namespace projectmetis::controller {
namespace {

using ::proto::ParseTextOrDie;
using ::testing::proto::EqualsProto;

const char kModel1[] = R"pb(
variables {
  name: "var1"
  trainable: true
  double_tensor {
    spec {
      length: 10
      dimensions: 5
      dimensions: 2
      dtype: DOUBLE
    }
  }
}
)pb";

TEST(FederatedAverageTest, CorrectAverage) {
  auto model1 = ParseTextOrDie<Model>(kModel1);
  auto model2 = ParseTextOrDie<Model>(kModel1);

  auto expected = ParseTextOrDie<Model>(kModel1);

  for (int i = 0; i < 10; ++i) {
    model1.mutable_variables(0)->mutable_double_tensor()->add_values(i);
    model2.mutable_variables(0)->mutable_double_tensor()->add_values(10 - i);

    expected.mutable_variables(0)->mutable_double_tensor()->add_values((i + (10 - i)) * 1.0 / 2.0);
  }

  std::vector to_aggregate({
      std::make_pair<const Model*, double>(&model1, 0.5),
      std::make_pair<const Model*, double>(&model2, 0.5)
  });

  FederatedAverage avg;
  FederatedModel averaged = avg.Aggregate(to_aggregate);

  EXPECT_THAT(averaged.model(), EqualsProto(expected));
}

} // namespace
} // namespace projectmetis::controller