load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")

package(default_visibility = ["//visibility:public"])

# Binaries.
cc_binary(
    name = "controller_main",
    srcs = ["controller_main.cc"],
    deps = [
        ":controller",
        ":controller_servicer",
        "@absl//absl/strings"
    ],
)

# Libraries.
cc_library(
    name = "controller",
    srcs = ["controller.cc"],
    hdrs = ["controller.h"],
    deps = [
        ":controller_utils",
        ":controller_cc_grpc_proto",
        "//projectmetis/controller/model_aggregation",
        "@absl//absl/status:statusor",
        "@absl//absl/container:flat_hash_map",
        "@absl//absl/memory",
    ],
)

cpp_grpc_library(
    name = "controller_cc_grpc_proto",
    protos = [
        "//projectmetis/proto:controller_proto",
        "//projectmetis/proto:core_proto",
    ]
)

cc_library(
    name = "controller_utils",
    srcs = ["controller_utils.cc"],
    hdrs = ["controller_utils.h"],
    deps = [
        ":controller_cc_grpc_proto",
        "@absl//absl/strings"
    ],
)

cc_library(
    name = "controller_servicer",
    srcs = ["controller_servicer.cc"],
    hdrs = ["controller_servicer.h"],
    deps = [
        ":controller",
        ":controller_cc_grpc_proto",
        "@absl//absl/memory",
    ],
)

# Tests.
cc_test(
    name = "controller_test",
    srcs = ["controller_test.cc"],
    deps = [
        ":controller",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ],
)

cc_test(
    name = "controller_servicer_test",
    srcs = ["controller_servicer_test.cc"],
    deps = [
        ":controller_servicer",
        "//projectmetis/core:macros",
        "//projectmetis/core/matchers:proto_matchers",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ],
)
