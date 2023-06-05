
#ifndef PROJECTMETIS_RC_PROJECTMETIS_CORE_MACROS_H_
#define PROJECTMETIS_RC_PROJECTMETIS_CORE_MACROS_H_

#include <string>

#include <google/protobuf/text_format.h>

#define CHECK(value) \
          do {           \
            if (value == false) { \
              throw std::runtime_error("Unable to load proto from text."); \
            }          \
          } while(0)

// Run a command that returns a util::Status.  If the called code returns an
// error status, return that status up out of this method too.
//
// Example:
//   RETURN_IF_ERROR(DoThings(4));
#define RETURN_IF_ERROR(expr)                                                \
  do {                                                                       \
    /* Using _status below to avoid capture problems if expr is "status". */ \
    ::absl::Status _status = (expr);              \
    if (ABSL_PREDICT_FALSE(!_status.ok())) return _status;               \
  } while (0)

namespace proto {
template<typename T>
T ParseTextOrDie(const std::string &input) {
  T result;
  CHECK(google::protobuf::TextFormat::ParseFromString(input, &result));
  return result;
}
}  // namespace proto

#endif //PROJECTMETIS_RC_PROJECTMETIS_CORE_MACROS_H_
