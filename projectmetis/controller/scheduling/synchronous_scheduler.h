
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_

#include "absl/container/flat_hash_set.h"
#include "projectmetis/controller/scheduling/scheduler.h"

namespace projectmetis::controller {

// Implements the synchronous task scheduling policy.
class SynchronousScheduler : public Scheduler {
 public:
  std::vector<std::string> ScheduleNext(const std::string &learner_id,
                                        const CompletedLearningTask &task,
                                        const std::vector<LearnerDescriptor> &active_learners) override {
    // First, it adds the learner id to the set.
    learner_ids_.insert(learner_id);

    // Second, it checks if the number of learners in the set is the same as
    // `total_num_learners_`.
    if (learner_ids_.size() != active_learners.size()) {
      // If not, then return an empty list. No need to schedule any task for the
      // moment.
      return std::vector<std::string>();
    }

    // Otherwise, schedule all learners for the next task.
    std::vector<std::string>
        to_schedule(learner_ids_.begin(), learner_ids_.end());

    // Clean the state.
    learner_ids_.clear();

    return to_schedule;
  }

  inline std::string name() override {
    return "SynchronousScheduler";
  }

 private:
  // Keeps track of the learners.
  ::absl::flat_hash_set<std::string> learner_ids_;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_
