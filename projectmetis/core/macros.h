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

namespace proto {
template<typename T>
T ParseTextOrDie(const std::string &input) {
  T result;
  CHECK(google::protobuf::TextFormat::ParseFromString(input, &result));
  return result;
}
}  // namespace proto

#endif //PROJECTMETIS_RC_PROJECTMETIS_CORE_MACROS_H_
