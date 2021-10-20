//
// MIT License
// 
// Copyright (c) 2021 Project Metis
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    if (learner_ids_.size() < active_learners.size()) {
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
    return "FedSync";
  }

 private:
  // Keeps track of the learners.
  ::absl::flat_hash_set<std::string> learner_ids_;
};

} // namespace projectmetis::controller

#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_SCHEDULING_SYNCHRONOUS_SCHEDULER_H_
