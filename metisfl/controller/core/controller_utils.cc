
#include <sys/resource.h>

#include "controller_utils.h"
#include "metisfl/controller/aggregation/model_aggregation.h"
#include "metisfl/controller/scaling/model_scaling.h"
#include "metisfl/controller/selection/model_selection.h"
#include "metisfl/controller/store/store.h"
#include "metisfl/controller/scheduling/scheduling.h"

namespace metisfl::controller {

std::unique_ptr<AggregationFunction>
CreateAggregator(const AggregationRule &aggregation_rule) {

  if (aggregation_rule.has_fed_avg()) {
    return absl::make_unique<FederatedAverage>();
  } else if (aggregation_rule.has_fed_rec()) {
    return absl::make_unique<FederatedRecency>();
  } else if (aggregation_rule.has_fed_stride()) {
    return absl::make_unique<FederatedStride>();
  } else if (aggregation_rule.has_pwa()) {
    return absl::make_unique<PWA>(aggregation_rule.pwa().he_scheme());
  } else {
    throw std::runtime_error("Unsupported aggregation rule.");
  }

}

std::unique_ptr<ModelStore>
CreateModelStore(const ModelStoreConfig &config) {

  if (config.has_in_memory_store()) {
    return absl::make_unique<HashMapModelStore>(config.in_memory_store());
  } else if (config.has_redis_db_store()) {
    return absl::make_unique<RedisModelStore>(config.redis_db_store());
  } else {
    throw std::runtime_error("Unsupported model store backend.");
  }

}

std::unique_ptr<ScalingFunction>
CreateScaler(const AggregationRuleSpecs &aggregation_rule_specs) {

  if (aggregation_rule_specs.scaling_factor() == AggregationRuleSpecs::NUM_COMPLETED_BATCHES) {
    return absl::make_unique<BatchesScaler>();
  } else if (aggregation_rule_specs.scaling_factor() == AggregationRuleSpecs::NUM_PARTICIPANTS) {
    return absl::make_unique<ParticipantsScaler>();
  } else if (aggregation_rule_specs.scaling_factor() == AggregationRuleSpecs::NUM_TRAINING_EXAMPLES) {
    return absl::make_unique<TrainDatasetSizeScaler>();
  } else {
    throw std::runtime_error("Unsupported scaler.");
  }

}

std::unique_ptr<Scheduler>
CreateScheduler(const CommunicationSpecs &specs) {

  if (specs.protocol() == CommunicationSpecs::SYNCHRONOUS ||
      specs.protocol() == CommunicationSpecs::SEMI_SYNCHRONOUS) {
    return absl::make_unique<SynchronousScheduler>();
  } else if (specs.protocol() == CommunicationSpecs::ASYNCHRONOUS) {
    return absl::make_unique<AsynchronousScheduler>();
  } else {
    throw std::runtime_error("Unsupported scheduling policy.");
  }

}

std::unique_ptr<Selector>
CreateSelector() {
  return absl::make_unique<ScheduledCardinality>();
}

long GetTotalMemory() {
  // This function records the entire process memory.
  // The memory size is ever-increasing. In other words,
  // if we free any resources then the memory does not reduce
  // but rather remains constant.
  // TODO(stripeli): We need a more fine-grained (dynamic) memory capture
  // tool that also accounts for memory release not just cumulative.
  struct rusage usage{};
  int ret = getrusage(RUSAGE_SELF, &usage);
  if (ret == 0) {
    // Capture value in kilobytes - maximum resident set size
    // utilized in KB. Metric value reflects the size of the
    // main and virtual memory of the parent process.
    return usage.ru_maxrss;
  } else {
    return 0;
  }
}

} // namespace metisfl::controller
