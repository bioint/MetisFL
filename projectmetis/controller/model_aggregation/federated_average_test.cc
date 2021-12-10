//
// MIT License
//
// Copyright (c) 2021 Project Metis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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