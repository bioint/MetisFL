
#include "controller_utils.h"

namespace metisfl::controller {

std::unique_ptr<AggregationFunction> CreateAggregator(
    const GlobalTrainParams &params, const DType_Type dtype) {
  const auto &aggregation_rule = params.aggregation_rule;

  if (aggregation_rule == "FedAvg")
    return CreateAggregatorForDType<FederatedAverage>(dtype);
  if (aggregation_rule == "FedRec")
    return CreateAggregatorForDType<FederatedRecency>(dtype);
  if (aggregation_rule == "FedStride")
    return CreateAggregatorForDType<FederatedStride>(dtype);
  if (aggregation_rule == "SecAgg") {
    return absl::make_unique<SecAgg>(params.he_batch_size,
                                     params.he_scaling_factor_bits,
                                     params.he_crypto_context_file);
  }
}

template <template <typename T> class C>
std::unique_ptr<AggregationFunction> CreateAggregatorForDType(
    DType_Type dtype) {
  switch (dtype) {
    case DType_Type_UINT8:
      return absl::make_unique<C<unsigned char>>();
    case DType_Type_UINT16:
      return absl::make_unique<C<unsigned short>>();
    case DType_Type_UINT32:
      return absl::make_unique<C<unsigned int>>();
    case DType_Type_UINT64:
      return absl::make_unique<C<unsigned long>>();
    case DType_Type_INT8:
      return absl::make_unique<C<signed char>>();
    case DType_Type_INT16:
      return absl::make_unique<C<signed short>>();
    case DType_Type_INT32:
      return absl::make_unique<C<signed int>>();
    case DType_Type_INT64:
      return absl::make_unique<C<signed long>>();
    case DType_Type_FLOAT32:
      return absl::make_unique<C<float>>();
    case DType_Type_FLOAT64:
      return absl::make_unique<C<double>>();
    default:
      PLOG(FATAL) << "Unsupported tensor data type.";
  }
}

std::unique_ptr<ModelStore> CreateModelStore(
    const ModelStoreParams &model_store_params) {
  const auto &model_store = model_store_params.model_store;
  const auto &lineage_length = model_store_params.lineage_length;
  const auto &model_store_hostname = model_store_params.hostname;
  const auto &model_store_port = model_store_params.port;
  if (model_store == "Redis")
    return absl::make_unique<RedisModelStore>(model_store_hostname,
                                              model_store_port, lineage_length);
  if (model_store == "InMemory")
    return absl::make_unique<HashMapModelStore>(lineage_length);

  PLOG(FATAL) << "Unsupported model store.";
}

std::unique_ptr<Scheduler> CreateScheduler(
    const std::string &communication_protocol) {
  if (communication_protocol == "Synchronous" ||
      communication_protocol == "SemiSynchronous")
    return absl::make_unique<SynchronousScheduler>();
  if (communication_protocol == "Asynchronous")
    return absl::make_unique<AsynchronousScheduler>();

  PLOG(FATAL) << "Unsupported communication protocol.";
}

std::unique_ptr<Selector> CreateSelector() {
  return absl::make_unique<ScheduledCardinality>();
}

std::string GenerateRadnomId() {
  std::string id = absl::StrCat(absl::ToUnixMicros(absl::Now()));
  return id;
}

long GetTotalMemory() {
  // This function records the entire process memory.
  // The memory size is ever-increasing. In other words,
  // if we free any resources then the memory does not reduce
  // but rather remains constant.
  // TODO(stripeli): We need a more fine-grained (dynamic) memory capture
  // tool that also accounts for memory release not just cumulative.
  struct rusage usage {};
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

}  // namespace metisfl::controller
