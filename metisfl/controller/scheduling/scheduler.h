
#ifndef METISFL_METISFL_CONTROLLER_SCHEDULING_SCHEDULER_H_
#define METISFL_METISFL_CONTROLLER_SCHEDULING_SCHEDULER_H_

#include <string>
#include <vector>

namespace metisfl::controller {

class Scheduler {
 protected:
  int global_iteration_ = 0;  // only makes sense for synchronous schedulers

 public:
  virtual ~Scheduler() = default;

  /**
   * @brief Schedule the next set of learners(s).
   *
   * @param learner_id The learner id of the learner that just finished.
   * @param num_active_learners The number of active learners.
   * @return std::vector<std::string> The next set of learners to schedule.
   */
  virtual std::vector<std::string> ScheduleNext(
      const std::string &learner_id, const int num_active_learners) = 0;

  virtual std::string name() = 0;

  /**
   * @brief Get the global iteration.
   *
   * @return int The global iteration.
   */
  int GetGlobalIteration() { return global_iteration_; }
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SCHEDULING_SCHEDULER_H_
