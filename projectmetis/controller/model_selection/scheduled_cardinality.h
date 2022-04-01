//
// MIT License
//
// Copyright (c) 2022 Project Metis
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
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SCHEDULEDCARDINALITY_H
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SCHEDULEDCARDINALITY_H

#include "projectmetis/controller/model_selection/selector.h"

namespace projectmetis::controller {

// A subset cardinality selector that picks the models that need to be
// considered during aggregation based on the cardinality of the learners
// subset (scheduled) collection.
class ScheduledCardinality : public Selector {
 public:

  std::vector<std::string> Select(
      const std::vector<std::string> &scheduled_learners,
      const std::vector<LearnerDescriptor> &active_learners) override {

    // The given set of scheduled learners needs to contain at
    // least 2 learners else we select all active learners.
    if (scheduled_learners.size() < 2) {
      std::vector<std::string> active_ids;
      for (const auto &learner_descriptor : active_learners) {
        active_ids.push_back(learner_descriptor.id());
      }
      return active_ids;
    } else {
      return scheduled_learners;
    }
  }

  std::string name() override {
    return "ScheduledCardinality";
  };
};

} // namespace projectmetis::controller


#endif //PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SCHEDULEDCARDINALITY_H
