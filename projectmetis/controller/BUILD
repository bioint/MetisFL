load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")

package(default_visibility = ["//visibility:public"])

cpp_grpc_library(
    name = "controller_cpp_grpc",
    protos = [
        "//projectmetis/proto:controller_proto",
        "//projectmetis/proto:core_proto",
    ]
)

cc_library(
    name = "controller_grpc",
    srcs = ["controller_service.cc"],
    hdrs = ["controller_service.h"],
    deps = [":controller_cpp_grpc"],
)

cc_test(
    name = "controller_grpc_test",
    srcs = ["controller_service_test.cc"],
    deps = [
        ":controller_grpc",
        "@gtest//:gtest",
        "@gtest//:gtest_main"
    ],
)

cc_binary(
    name = "controller",
    srcs = ["controller_server.cc"],
    deps = [":controller_grpc"],
)
