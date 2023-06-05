
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "absl/strings/str_cat.h"
#include "projectmetis/controller/scheduling/synchronous_scheduler.h"

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
TEST(SynchronousScheduler, SingleLearner) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(1);

  auto res =
      scheduler.ScheduleNext("learner1", CompletedLearningTask(), learners);
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0], "learner1");
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, TwoLearners) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(2);

  auto res1 =
      scheduler.ScheduleNext("learner1", CompletedLearningTask(), learners);
  EXPECT_TRUE(res1.empty());

  auto res2 =
      scheduler.ScheduleNext("learner2", CompletedLearningTask(), learners);
  EXPECT_THAT(res2, UnorderedElementsAre("learner1", "learner2"));
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, MultipleLearners) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (int i = 0; i < 4; ++i) {
    const auto &learner = learners[i];
    scheduler.ScheduleNext(learner.id(), CompletedLearningTask(), learners);
  }

  auto res =
      scheduler.ScheduleNext("learner5", CompletedLearningTask(), learners);
  EXPECT_THAT(res,
              UnorderedElementsAre("learner1",
                                   "learner2",
                                   "learner3",
                                   "learner4",
                                   "learner5"));
}

// NOLINTNEXTLINE
TEST(SynchronousScheduler, NoDoubleSchedule) {
  SynchronousScheduler scheduler;
  auto learners = CreateLearners(5);

  for (const auto &learner : learners) {
    scheduler.ScheduleNext(learner.id(), CompletedLearningTask(), learners);
  }

  auto res = scheduler.ScheduleNext("learner1", CompletedLearningTask(), learners);
  EXPECT_TRUE(res.empty());
}

} // namespace
} // namespace projectmetis::controller