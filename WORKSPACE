workspace(name = "metisfl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")


###### PyBind11
http_archive(
  name = "pybind11_bazel",
  sha256 = "3700ef34597cda8e9fa4a43a6f177bec87d6d07fb400f3dfdf636a71332282e1",
  urls = ["https://github.com/raschild/pybind11_bazel/archive/refs/heads/main.zip"],
  strip_prefix = "pybind11_bazel-main",
)

_PYBIND11_VERSION_ = "2.8.0"

http_archive(
  name = "pybind11",
  sha256 = "9ca7770fc5453b10b00a4a2f99754d7a29af8952330be5f5602e7c2635fa3e79",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-" + _PYBIND11_VERSION_,
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v" + _PYBIND11_VERSION_ + ".tar.gz"],
)

###### ABSL
_ABSL_VERSION_ = "20210324.2"

http_archive(
    name = "absl",
    sha256 = "59b862f50e710277f8ede96f083a5bb8d7c9595376146838b9580be90374ee1f",
    strip_prefix = "abseil-cpp-" + _ABSL_VERSION_,
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/" + _ABSL_VERSION_ + ".tar.gz"],
)


###### GTest
_GTEST_VERSION_ = "1.10.0"

http_archive(
    name = "gtest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-" + _GTEST_VERSION_,
    urls = ["https://github.com/google/googletest/archive/release-" + _GTEST_VERSION_ + ".tar.gz"],
)

###### GRPC
_GRPC_VERSION_ = "3.1.1"

http_archive(
    name = "rules_proto_grpc",
    sha256 = "7954abbb6898830cd10ac9714fbcacf092299fda00ed2baf781172f545120419",
    strip_prefix = "rules_proto_grpc-" + _GRPC_VERSION_,
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/" + _GRPC_VERSION_ + ".tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

load("@rules_proto_grpc//cpp:repositories.bzl", rules_proto_grpc_cpp_repos = "cpp_repos")
rules_proto_grpc_cpp_repos()

load("@rules_proto_grpc//python:repositories.bzl", rules_proto_grpc_py_repos = "python_repos")
rules_proto_grpc_py_repos()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()


###### PALISADE
new_git_repository(
    name = "palisade_git",
    remote = "https://gitlab.com/palisade/palisade-release.git",
    commit = "19e84ae53ecdd1ca3360c4cdbb6df2eb31f21195",
    init_submodules = True,
    shallow_since = "1651346647 +0000",
    verbose = True,
    recursive_init_submodules = True,
    build_file_content = """
filegroup(
    name = "palisade_srcs",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)

###### FOREIGN_RULES_CC
_FOREIGN_RULES_VERSION_ = "0.8.0"

http_archive(
    name = "rules_foreign_cc",
    sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
    strip_prefix = "rules_foreign_cc-" + _FOREIGN_RULES_VERSION_,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/" + _FOREIGN_RULES_VERSION_ + ".tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.8.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()
###### GLOG
# Use Google Logging (glog) to implement application-level logging at the controller.
_GFLAG_VERSION_ = "2.2.2"
_GLOG_VERSION_ = "0.6.0"

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-" + _GFLAG_VERSION_,
    url = "https://github.com/gflags/gflags/archive/v" + _GFLAG_VERSION_ + ".tar.gz",
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-" + _GLOG_VERSION_,
    url = "https://github.com/google/glog/archive/v" + _GLOG_VERSION_ + ".zip",
)


###### REDIS
_HIREDIS_VERSION_ = "1.0.2"

http_archive(
    name = "hiredis_git",
    sha256 = "e0ab696e2f07deb4252dda45b703d09854e53b9703c7d52182ce5a22616c3819",
    strip_prefix = "hiredis-" + _HIREDIS_VERSION_,
    url = "https://github.com/redis/hiredis/archive/refs/tags/v" + _HIREDIS_VERSION_ + ".tar.gz",
    build_file_content =
"""
# The build file's content is derived from
# https://github.com/ray-project/ray/blob/master/bazel/BUILD.hiredis
# This library is for internal hiredis use, because hiredis assumes a
# different include prefix for itself than external libraries do.
cc_library(
    name = "_hiredis",
    hdrs = [
        "dict.c",
    ],
)

cc_library(
    name = "hiredis",
    srcs = glob(
        [
            "*.c",
            "*.h",
        ],
        exclude =
        [
            "ssl.c",
            "test.c",
        ],
    ),
    hdrs = glob([
        "*.h",
        "adapters/*.h",
    ]),
    include_prefix = "hiredis",
    deps = [
        ":_hiredis",
    ],
    visibility = ["//visibility:public"],
)
""",
)
