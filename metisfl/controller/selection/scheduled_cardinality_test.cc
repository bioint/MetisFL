
#include "metisfl/controller/selection/scheduled_cardinality.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "absl/strings/str_cat.h"

namespace metisfl::controller {
namespace {

using ::testing::Return;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;

std::vector<std ::string> CreateLearners(int n) {
  std::vector<std::string> learner_ids;
  for (int i = 0; i < n; ++i) {
    learner_ids.emplace_back(absl::StrCat("learner", i));
  }
  return learner_ids;
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
  for (const auto &id : active_learners) {
    scheduled_learners.emplace_back(id);
  }

  ScheduledCardinality selector;
  auto res = selector.Select(scheduled_learners, active_learners);
  ASSERT_EQ(res.size(), 5);
}

}  // namespace
}  // namespace metisfl::controller
