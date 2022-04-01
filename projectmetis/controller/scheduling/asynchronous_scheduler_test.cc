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

#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "absl/strings/str_cat.h"
#include "projectmetis/controller/scheduling/asynchronous_scheduler.h"

namespace projectmetis::controller {
namespace {

using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::Return;

std::vector<LearnerDescriptor> CreateLearners(int n) {
  std::vector<LearnerDescriptor> learners;
  for (int i = 0; i < n; ++i) {
    LearnerDescriptor learner;
    learner.set_id(absl::StrCat("learner", i + 1));
    learners.push_back(learner);
  }
  return learners;
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, SingleLearner) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(1);

  auto res =
      scheduler.ScheduleNext("learner1", CompletedLearningTask(), learners);
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner1");
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, TwoLearners) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(2);

  auto res1 =
      scheduler.ScheduleNext("learner1", CompletedLearningTask(), learners);
  ASSERT_EQ(res1.size(), 1);
  EXPECT_EQ(res1[0], "learner1");

  auto res2 =
      scheduler.ScheduleNext("learner2", CompletedLearningTask(), learners);
  ASSERT_EQ(res2.size(), 1);
  EXPECT_EQ(res2[0], "learner2");
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, MultipleLearners) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (int i = 0; i < 4; ++i) {
    const auto &learner = learners[i];
    scheduler.ScheduleNext(learner.id(), CompletedLearningTask(), learners);
  }

  auto res =
      scheduler.ScheduleNext("learner5", CompletedLearningTask(), learners);
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner5");

}

} // namespace
} // namespace projectmetis::controller
