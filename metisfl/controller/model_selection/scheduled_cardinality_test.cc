
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "absl/strings/str_cat.h"
#include "metisfl/controller/model_selection/scheduled_cardinality.h"

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
TEST(ScheduledCardinality, NoLearner) {
  auto active_learners = CreateLearners(5);
  std::vector<std::string> scheduled_learners;

  ScheduledCardinality selector;
  auto res = selector.Select(scheduled_learners, active_learners);
  ASSERT_EQ(res.size(), 5);
}

// NOLINTNEXTLINE
TEST(ScheduledCardinality, SingleLearner) {
  auto active_learners = CreateLearners(5);
  std::vector<std::string> scheduled_learners;
  scheduled_learners.emplace_back("learner1");

  ScheduledCardinality selector;
  auto res = selector.Select(scheduled_learners, active_learners);
  ASSERT_EQ(res.size(), 5);
}

// NOLINTNEXTLINE
TEST(ScheduledCardinality, TwoLearners) {
  auto active_learners = CreateLearners(5);
  std::vector<std::string> scheduled_learners;
  scheduled_learners.emplace_back("learner1");
  scheduled_learners.emplace_back("learner2");

  ScheduledCardinality selector;
  auto res = selector.Select(scheduled_learners, active_learners);
  ASSERT_EQ(res.size(), 2);
}

// NOLINTNEXTLINE
TEST(ScheduledCardinality, AllLearners) {
  auto active_learners = CreateLearners(5);
  std::vector<std::string> scheduled_learners;
  for (const auto &learner_descriptor : active_learners) {
    scheduled_learners.emplace_back(learner_descriptor.id());
  }

  ScheduledCardinality selector;
  auto res = selector.Select(scheduled_learners, active_learners);
  ASSERT_EQ(res.size(), 5);
}

} // namespace
} // namespace projectmetis::controller
