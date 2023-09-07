
#ifndef METISFL_METISFL_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_
#define METISFL_METISFL_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_

#include <glog/logging.h>

#include "absl/container/flat_hash_set.h"
#include "metisfl/controller/scheduling/scheduler.h"

namespace metisfl::controller {

class SynchronousScheduler : public Scheduler {
 public:
  std::vector<std::string> ScheduleNext(
      const std::string &learner_id, const int num_active_learners) override {
    learner_ids_.insert(learner_id);

    if (learner_ids_.size() != num_active_learners) {
      return {};
    }

    std::vector<std::string> to_schedule(learner_ids_.begin(),
                                         learner_ids_.end());
    learner_ids_.clear();
    ++global_iteration_;

    LOG(INFO) << "Starting Federation Round " << global_iteration_;

    return to_schedule;
  }

  inline std::string name() override { return "SynchronousScheduler"; }

 private:
  absl::flat_hash_set<std::string> learner_ids_;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_
