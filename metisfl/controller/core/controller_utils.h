
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_

#include <fstream> // std::ifstream
#include <sstream> // std::stringstream
#include <string> // std::string

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "metisfl/proto/metis.pb.h"
#include "metisfl/controller/aggregation/model_aggregation.h"
#include "metisfl/controller/scaling/model_scaling.h"
#include "metisfl/controller/selection/model_selection.h"
#include "metisfl/controller/store/store.h"
#include "metisfl/controller/scheduling/scheduling.h"

namespace metisfl::controller {

std::unique_ptr<AggregationFunction>
CreateAggregator(const AggregationRule &aggregation_rule);

std::unique_ptr<ModelStore>
CreateModelStore(const ModelStoreConfig &config);

std::unique_ptr<ScalingFunction>
CreateScaler(const AggregationRuleSpecs &aggregation_rule_specs);

std::unique_ptr<Scheduler>
CreateScheduler(const CommunicationSpecs &specs);

std::unique_ptr<Selector>
CreateSelector();

long GetTotalMemory();

// Generates a unique identifier for the provided learner entity. In the current
// implementation, the generated id is in the format of `<hostname>:<port>`.
inline std::string GenerateLearnerId(const ServerEntity &server_entity) {
  return absl::StrCat(server_entity.hostname(), ":", server_entity.port());
}

// Reads a file from disk. Error is raised if the file cannot be opened.
inline int ReadParseFile(std::string &file_content, std::string &file_name) {
  std::ifstream _file;
  _file.open(file_name);

  // Manage handling in case the stream buffer cannot be generated.
  std::stringstream buffer;
  if (_file.is_open()) {
    buffer << _file.rdbuf();
    file_content = buffer.str();
    return 1;
  }
  return -1;
}

} // namespace metisfl::controller

#endif //METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
