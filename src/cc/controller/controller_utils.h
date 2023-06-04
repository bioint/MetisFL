
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_

#include <fstream> // std::ifstream
#include <sstream> // std::stringstream
#include <string> // std::string

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/proto/metis.pb.h"
#include "src/cc/controller/model_aggregation/model_aggregation.h"
#include "src/cc/controller/model_scaling/model_scaling.h"
#include "src/cc/controller/model_selection/model_selection.h"
#include "src/cc/controller/model_storing/model_storing.h"
#include "src/cc/controller/scheduling/scheduling.h"

namespace projectmetis::controller {

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

// Reads a file from disk containing the key and certificate information
// and returns the certificate and a referenced argument. Error is raised
// if the file cannot be opened.
inline int ReadParseFile(std::string &file_content, std::string &file_name) {
  std::ifstream _file;
  _file.open(file_name);

  // Manage handling in case the certificates are not generated.
  std::stringstream buffer;
  if (_file.is_open()) {
    buffer << _file.rdbuf();
    file_content = buffer.str();
    return 1;
  }
  return -1;
}

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
