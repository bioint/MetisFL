
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

  virtual Model Aggregate(
      std::vector<std::vector<std::pair<const Model *, double>>> &pairs) = 0;

  inline virtual int RequiredLearnerLineageLength() const = 0;

  inline virtual std::string Name() const;

  virtual void Reset() = 0;
};

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_AGGREGATION_AGGREGATION_FUNCTION_H_
