
#ifndef METISFL_METISFL_CONTROLLER_COMMON_MACROS_H_
#define METISFL_METISFL_CONTROLLER_COMMON_MACROS_H_

#include <google/protobuf/text_format.h>

#include <string>

#define VALIDATE(value)                                            \
  do {                                                             \
    if (value == false) {                                          \
      throw std::runtime_error("Unable to load proto from text."); \
    }                                                              \
  } while (0)

// Run a command that returns a util::Status.  If the called code returns an
// error status, return that status up out of this method too.
//
// Example:
//   RETURN_IF_ERROR(DoThings(4));
#define RETURN_IF_ERROR(expr)                                                \
  do {                                                                       \
    /* Using _status below to avoid capture problems if expr is "status". */ \
    ::absl::Status _status = (expr);                                         \
    if (ABSL_PREDICT_FALSE(!_status.ok())) return _status;                   \
  } while (0)

namespace proto {
template <typename T>
T ParseTextOrDie(const std::string &input) {
  T result;
  VALIDATE(google::protobuf::TextFormat::ParseFromString(input, &result));
  return result;
}
}  // namespace proto

#endif  // METISFL_METISFL_CONTROLLER_COMMON_MACROS_H_
