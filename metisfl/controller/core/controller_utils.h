
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_

#include <sys/resource.h>

#include <fstream>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "metisfl/controller/aggregation/aggregation.h"
#include "metisfl/controller/common/proto_tensor_serde.h"
#include "metisfl/controller/core/types.h"
#include "metisfl/controller/scheduling/scheduling.h"
#include "metisfl/controller/selection/selection.h"
#include "metisfl/controller/store/store.h"

namespace metisfl::controller {

std::unique_ptr<AggregationFunction> CreateAggregator(
    const GlobalTrainParams &global_train_params);

std::unique_ptr<ModelStore> CreateModelStore(
    const ModelStoreParams &model_store_params);

std::unique_ptr<Scheduler> CreateScheduler(
    const std::string &scheduler);

std::unique_ptr<Selector> CreateSelector();

long GetTotalMemory();

inline std::string GenerateLearnerId(const std::string &hostname,
                                     const int port) {
  return absl::StrCat(hostname, ":", port);
}

std::string GenerateRadnomId();

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
