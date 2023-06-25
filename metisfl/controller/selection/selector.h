
#ifndef METISFL_METISFL_CONTROLLER_SELECTION_SELECTOR_H_
#define METISFL_METISFL_CONTROLLER_SELECTION_SELECTOR_H_

#include "metisfl/proto/metis.pb.h"

namespace metisfl::controller {

// A selector picks the models that need to be considered during aggregation.
class Selector {
 public:
  virtual ~Selector() = default;

  // Returns the ids of the learners that need to be selected from the
  // collections of scheduled learners, which refers to learners that
  // will run the next training task and active learners, which refers to
  // learners that are currently part of the federation.
  virtual std::vector<std::string> Select(
      const std::vector<std::string> &scheduled_learners,
      const std::vector<LearnerDescriptor> &active_learners) = 0;

  virtual std::string name() = 0;
};

} // namespace metisfl::controller


#endif //METISFL_METISFL_CONTROLLER_SELECTION_SELECTOR_H_
