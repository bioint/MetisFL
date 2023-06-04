
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_

#include "src/cc/controller/scheduling/scheduler.h"

namespace projectmetis::controller {

// Implements the asynchronous task scheduling policy.
class AsynchronousScheduler : public Scheduler {
 public:
  std::vector<std::string> ScheduleNext(const std::string &learner_id,
                                        const CompletedLearningTask &task,
                                        const std::vector<LearnerDescriptor> &active_learners) override {

    // Schedules current learner for its next task.
    return std::vector<std::string> {learner_id};

  }

  inline std::string name() override {
    return "AsynchronousScheduler";
  }

};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
