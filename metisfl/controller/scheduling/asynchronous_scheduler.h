
#ifndef METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
#define METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_

#include "metisfl/controller/scheduling/scheduler.h"

namespace metisfl::controller {

class AsynchronousScheduler : public Scheduler {
 public:
  std::vector<std::string> ScheduleNext(
      const std::string &learner_id, const int num_active_learners) override {
    return std::vector<std::string>{learner_id};
  }

  inline std::string name() override { return "AsynchronousScheduler"; }

 private:
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCHEDULING_ASYNCHRONOUS_SCHEDULER_H_
