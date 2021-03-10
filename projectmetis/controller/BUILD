load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")

package(default_visibility = ["//visibility:public"])

cpp_grpc_library(
    name = "controller_cpp_grpc",
    protos = [
        "//projectmetis/controller/protos:controller_proto",
        "//projectmetis/core/protos:core_proto",
    ]
)

cc_library(
    name = "controller_grpc",
    srcs = ["controller_service.cc"],
    hdrs = ["controller_service.h"],
    deps = [":controller_cpp_grpc"],
)

cc_binary(
    name = "controller",
    srcs = ["main.cc"],
    deps = [
        ":controller_grpc"
    ],
)
