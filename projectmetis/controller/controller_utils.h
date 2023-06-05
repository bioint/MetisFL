
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_

#include <string>

#include "absl/strings/str_cat.h"
#include "projectmetis/proto/metis.pb.h"

namespace projectmetis::controller {

// Generates a unique identifier for the provided learner entity. In the current
// implementation, the generated id is in the format of `<hostname>:<port>`.
inline std::string GenerateLearnerId(const ServerEntity &server_entity) {
  return absl::StrCat(server_entity.hostname(), ":", server_entity.port());
}

} // namespace projectmetis::controller

#endif // PROJECTMETIS_RC_PROJECTMETIS_CONTROLLER_CONTROLLER_UTILS_H_
