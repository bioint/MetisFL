
#include "metisfl/controller/scheduling/asynchronous_scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "absl/strings/str_cat.h"
#include "metisfl/proto/controller.pb.h"

namespace metisfl::controller {
namespace {

using ::testing::Return;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

std::vector<std::string> CreateLearners(int n) {
  std::vector<std::string> learners;
  for (int i = 0; i < n; ++i) {
    learners.push_back(absl::StrCat("learner", i + 1));
  }
  return learners;
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, SingleLearner) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(1);

  auto res = scheduler.ScheduleNext("learner1", learners.size());
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner1");
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, TwoLearners) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(2);

  auto res1 = scheduler.ScheduleNext("learner1", learners.size());
  ASSERT_EQ(res1.size(), 1);
  EXPECT_EQ(res1[0], "learner1");

  auto res2 = scheduler.ScheduleNext("learner2", learners.size());
  ASSERT_EQ(res2.size(), 1);
  EXPECT_EQ(res2[0], "learner2");
}

// NOLINTNEXTLINE
TEST(AsynchronousScheduler, MultipleLearners) {
  AsynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (int i = 0; i < 4; ++i) {
    const auto &learner = learners[i];
    scheduler.ScheduleNext(learner, learners.size());
  }

  auto res = scheduler.ScheduleNext("learner5", learners.size());
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner5");
}

}  // namespace
}  // namespace metisfl::controller
