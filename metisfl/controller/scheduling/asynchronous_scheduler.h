
#ifndef METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
#define METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_

#include "metisfl/controller/scheduling/scheduler.h"

namespace metisfl::controller {

// Implements the asynchronous task scheduling policy.
class AsynchronousScheduler : public Scheduler {
 public:
  std::vector<std::string> ScheduleNext(
      const std::string &learner_id,
      const int num_active_learners) override {
    // Schedules current learner for its next task.
    return std::vector<std::string>{learner_id};
  }

  inline std::string name() override { return "AsynchronousScheduler"; }
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
