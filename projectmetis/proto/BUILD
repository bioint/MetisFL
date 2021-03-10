load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "core_proto",
    srcs = [
        "optimizers.proto",
        "federatedmodel.proto",
        "shared.proto"
    ],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
    ],
)

proto_library(
    name = "controller_proto",
    srcs = ["controller.proto"],
    deps = [":core_proto"],
)

proto_library(
    name = "learner_proto",
    srcs = ["learner.proto"],
    deps = [":core_proto"],
)