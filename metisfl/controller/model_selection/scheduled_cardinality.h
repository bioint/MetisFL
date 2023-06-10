
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SCHEDULEDCARDINALITY_H
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_MODEL_SELECTION_SCHEDULEDCARDINALITY_H

#include "metisfl/controller/model_selection/selector.h"

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
