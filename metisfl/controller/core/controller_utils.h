
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_

#include <sys/resource.h>

#include <fstream>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "metisfl/controller/aggregation/aggregation.h"
#include "metisfl/controller/scaling/scaling.h"
#include "metisfl/controller/scheduling/scheduling.h"
#include "metisfl/controller/selection/selection.h"
#include "metisfl/controller/store/store.h"

typedef struct ServerParams {
  std::string hostname;
  int port;
  std::string public_certificate;
  std::string private_key;
  std::string root_certificate;
} ServerParams;

typedef struct GlobalTrainParams {
  std::string aggregation_rule;
  std::string communication_protocol;
  std::string scaling_factor;
  float participation_ratio;
  int stride_length;

  int he_batch_size;
  int he_scaling_factor_bits;
  std::string he_crypto_context_file;

  float semi_sync_lambda;
  int semi_sync_recompute_num_updates;
} GlobalTrainParams;

typedef struct ModelStoreParams {
  std::string model_store;
  int lineage_length;
  std::string hostname;
  int port;
} ModelStoreParams;

namespace metisfl::controller {

std::unique_ptr<AggregationFunction> CreateAggregator(
    const GlobalTrainParams &global_train_params);

std::unique_ptr<ModelStore> CreateModelStore(
    const ModelStoreParams &model_store_params);

std::unique_ptr<ScalingFunction> CreateScaler(
    const std::string &scaling_factor);

std::unique_ptr<Scheduler> CreateScheduler(
    const std::string &communication_protocol);

std::unique_ptr<Selector> CreateSelector();

long GetTotalMemory();

inline std::string GenerateLearnerId(const std::string &hostname,
                                     const int port) {
  return absl::StrCat(hostname, ":", port);
}

inline int ReadParseFile(std::string &file_content,
                         const std::string &file_name) {
  std::ifstream _file;
  _file.open(file_name);

  std::stringstream buffer;
  if (_file.is_open()) {
    buffer << _file.rdbuf();
    file_content = buffer.str();
    return 1;
  }
  return -1;
}

}  // namespace metisfl::controller

#endif  // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
