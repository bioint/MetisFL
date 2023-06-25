
#ifndef METISFL_METISFL_CONTROLLER_COMMON_PROTO_MATCHERS_H_
#define METISFL_METISFL_CONTROLLER_COMMON_PROTO_MATCHERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <google/protobuf/util/message_differencer.h>

namespace testing::proto {
using ::google::protobuf::util::MessageDifferencer;

MATCHER_P(EquivToProto, expected, "EquivToProto") {
  // Equivalence requires that the two messages have the same descriptor and
  // having same value. If expected has default values then equivalency holds.
  return MessageDifferencer::Equivalent(arg, expected);
}

MATCHER_P(EqualsProto, expected, "EqualsProto") {
  return MessageDifferencer::Equals(arg, expected);
}

MATCHER_P(ApproximatelyEquals, expected, "ApproximatelyEquals") {
  return MessageDifferencer::ApproximatelyEquals(arg, expected);
}

}  // namespace testing::proto

#endif //METISFL_METISFL_CONTROLLER_COMMON_PROTO_MATCHERS_H_
