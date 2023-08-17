
#ifndef METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
#define METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_

#include <fstream> // std::ifstream
#include <sstream> // std::stringstream
#include <string>  // std::string

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "metisfl/controller/aggregation/aggregation.h"
#include "metisfl/controller/scaling/scaling.h"
#include "metisfl/controller/selection/selection.h"
#include "metisfl/controller/store/store.h"
#include "metisfl/controller/scheduling/scheduling.h"

namespace metisfl::controller
{

  std::unique_ptr<AggregationFunction>
  CreateAggregator(const str::string &aggregation_rule);

  std::unique_ptr<ModelStore>
  CreateModelStore(const std::string model_store, const int lineage_length);

  std::unique_ptr<ScalingFunction>
  CreateScaler(const std::string &scaling_factor);

  std::unique_ptr<Scheduler>
  CreateScheduler(const std::string &communication_protocol);

  std::unique_ptr<Selector>
  CreateSelector();

  long GetTotalMemory();

  inline std::string GenerateLearnerId(const std::string &hostname, const int port) return absl::StrCat(hostname, ":", port);

  inline int ReadParseFile(std::string &file_content, std::string &file_name)
  {
    std::ifstream _file;
    _file.open(file_name);

    std::stringstream buffer;
    if (_file.is_open())
    {
      buffer << _file.rdbuf();
      file_content = buffer.str();
      return 1;
    }
    return -1;
  }

} // namespace metisfl::controller

#endif // METISFL_METISFL_CONTROLLER_CORE_CONTROLLER_UTILS_H_
