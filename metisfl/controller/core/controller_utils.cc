
#include <sys/resource.h>

#include "controller_utils.h"
#include "metisfl/controller/aggregation/aggregation.h"
#include "metisfl/controller/scaling/scaling.h"
#include "metisfl/controller/selection/selection.h"
#include "metisfl/controller/store/store.h"
#include "metisfl/controller/scheduling/scheduling.h"

namespace metisfl::controller
{

  std::unique_ptr<AggregationFunction>
  CreateAggregator(const std::string &aggregation_rule)
  {
    switch (aggregation_rule)
    {
    case "FedAvg":
      return absl::make_unique<FederatedAverage>();
    case "FedRec":
      return absl::make_unique<FederatedRecency>();
    case "FedStride":
      return absl::make_unique<FederatedStride>();
    case "SecAgg":
      return absl::make_unique<SecAgg>();
    default:
      throw std::runtime_error("Unsupported aggregation rule.");
    }
  }

  std::unique_ptr<ModelStore>
  CreateModelStore(const std::string model_store, const int lineage_length)
  {
    switch (model_store)
    {
    case "Redis":
      return absl::make_unique<RedisModelStore>(lineage_length);
    case "InMemory":
      return absl::make_unique<HashMapModelStore>(lineage_length);
    default:
      throw std::runtime_error("Unsupported model store backend.");
    }
  }

  std::unique_ptr<ScalingFunction>
  CreateScaler(const std::string &scaling_factor)
  {
    switch (scaling_factor)
    {
    case "NumCompletedBatches":
      return absl::make_unique<BatchesScaler>();
    case "NumParticipants":
      return absl::make_unique<ParticipantsScaler>();
    case "NumTrainingExamples":
      return absl::make_unique<TrainDatasetSizeScaler>();
    default:
      throw std::runtime_error("Unsupported scaler.");
    }
  }

  std::unique_ptr<Scheduler>
  CreateScheduler(const std::string &communication_protocol)
  {
    switch (communication_protocol)
    {
    case "Synchronous":
      return absl::make_unique<SynchronousScheduler>();
    case "SemiSynchronous":
      return absl::make_unique<SynchronousScheduler>();
    case "Asynchronous":
      return absl::make_unique<AsynchronousScheduler>();
    default:
      throw std::runtime_error("Unsupported scheduling policy.");
    }
  }

  std::unique_ptr<Selector>
  CreateSelector()
  {
    return absl::make_unique<ScheduledCardinality>();
  }

  long GetTotalMemory()
  {
    // This function records the entire process memory.
    // The memory size is ever-increasing. In other words,
    // if we free any resources then the memory does not reduce
    // but rather remains constant.
    // TODO(stripeli): We need a more fine-grained (dynamic) memory capture
    // tool that also accounts for memory release not just cumulative.
    struct rusage usage
    {
    };
    int ret = getrusage(RUSAGE_SELF, &usage);
    if (ret == 0)
    {
      // Capture value in kilobytes - maximum resident set size
      // utilized in KB. Metric value reflects the size of the
      // main and virtual memory of the parent process.
      return usage.ru_maxrss;
    }
    else
    {
      return 0;
    }
  }

} // namespace metisfl::controller
