
#include "metisfl/controller/scheduling/synchronous_scheduler.h"

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
TEST(SynchronousScheduler, SingleLearner) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(1);

  auto res = scheduler.ScheduleNext("learner1", learners.size());
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner1");
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, TwoLearners) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(2);

  auto res1 = scheduler.ScheduleNext("learner1", learners.size());
  EXPECT_TRUE(res1.empty());

  auto res2 = scheduler.ScheduleNext("learner2", learners.size());
  EXPECT_THAT(res2, UnorderedElementsAre("learner1", "learner2"));
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, MultipleLearners) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (int i = 0; i < 4; ++i) {
    const auto &learner = learners[i];
    scheduler.ScheduleNext(learner, learners.size());
  }

  auto res = scheduler.ScheduleNext("learner5", learners.size());
  EXPECT_THAT(res, UnorderedElementsAre("learner1", "learner2", "learner3",
                                        "learner4", "learner5"));
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, NoDoubleSchedule) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (const auto &learner : learners) {
    scheduler.ScheduleNext(learner, learners.size());
  }

  auto res = scheduler.ScheduleNext("learner1", learners.size());
  EXPECT_TRUE(res.empty());
}

}  // namespace
}  // namespace metisfl::controller
