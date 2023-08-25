
#ifndef METISFL_METISFL_CONTROLLER_SELECTION_SCHEDULED_CARDINALITY_H
#define METISFL_METISFL_CONTROLLER_SELECTION_SCHEDULED_CARDINALITY_H

#include "metisfl/controller/selection/selector.h"

namespace metisfl::controller {

// A subset cardinality selector that picks the models that need to be
// considered during aggregation based on the cardinality of the learners
// subset (scheduled) collection.
class ScheduledCardinality : public Selector {
 public:
  std::vector<std::string> Select(
      const std::vector<std::string> &scheduled_learners,
      const std::vector<std::string> &active_learners) override {
    // The given set of scheduled learners needs to contain at
    // least 2 learners else we select all active learners.
    if (scheduled_learners.size() < 2) {
      return active_learners;
    } else {
      return scheduled_learners;
    }
  }

  std::string name() override { return "ScheduledCardinality"; };
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_SELECTION_SCHEDULED_CARDINALITY_H
