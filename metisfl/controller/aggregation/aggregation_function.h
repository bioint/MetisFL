
#ifndef METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_
#define METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_

#include <omp.h>

#include <utility>
#include <vector>

#include "metisfl/proto/model.pb.h"

namespace metisfl::controller {

class AggregationFunction {
 public:
  virtual ~AggregationFunction() = default;

  virtual FederatedModel Aggregate(
      std::vector<std::vector<std::pair<Model *, double>>> &pairs) = 0;

  [[nodiscard]] inline virtual int RequiredLearnerLineageLength() const = 0;

  [[nodiscard]] inline virtual std::string Name() const = 0;

  virtual void Reset() = 0;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_
