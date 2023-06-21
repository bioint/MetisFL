workspace(name = "metisfl")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Imports PyBind11 Bazel plugin.
http_archive(
  name = "pybind11_bazel",
  sha256 = "3700ef34597cda8e9fa4a43a6f177bec87d6d07fb400f3dfdf636a71332282e1",
  urls = ["https://github.com/raschild/pybind11_bazel/archive/refs/heads/main.zip"],
  strip_prefix = "pybind11_bazel-main",
)

# Imports PyBind11 library.
http_archive(
  name = "pybind11",
  sha256 = "9ca7770fc5453b10b00a4a2f99754d7a29af8952330be5f5602e7c2635fa3e79",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.8.0",
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.8.0.tar.gz"],
)

# Imports Python rules.
http_archive(
    name = "rules_python",
    sha256 = "a644da969b6824cc87f8fe7b18101a8a6c57da5db39caa6566ec6109f37d2141",
    strip_prefix = "rules_python-0.20.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.20.0/rules_python-0.20.0.tar.gz",
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")


# Imports abseil.
http_archive(
    name = "absl",
    sha256 = "59b862f50e710277f8ede96f083a5bb8d7c9595376146838b9580be90374ee1f",
    strip_prefix = "abseil-cpp-20210324.2",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20210324.2.tar.gz"],
)

# Imports googletest.
http_archive(
    name = "gtest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = ["https://github.com/google/googletest/archive/release-1.10.0.tar.gz"],
)

# Imports grpc.
http_archive(
    name = "rules_proto_grpc",
    sha256 = "7954abbb6898830cd10ac9714fbcacf092299fda00ed2baf781172f545120419",
    strip_prefix = "rules_proto_grpc-3.1.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/3.1.1.tar.gz"],
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

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# Import new git repo rule
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Imports Palisades library.
new_git_repository(
    name = "palisade_git",
    remote = "https://gitlab.com/palisade/palisade-release.git",
    # tag: v1.11.7
    commit = "19e84ae53ecdd1ca3360c4cdbb6df2eb31f21195",
    strip_prefix = "",
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

http_archive(
    name = "rules_foreign_cc",
    sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
    strip_prefix = "rules_foreign_cc-0.8.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.8.0.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.8.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()

# Use Google Logging (glog) to implement application-level logging at the controller.
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    url = "https://github.com/google/glog/archive/v0.6.0.zip",
)

http_archive(
    name = "hiredis_git",
    sha256 = "e0ab696e2f07deb4252dda45b703d09854e53b9703c7d52182ce5a22616c3819",
    strip_prefix = "hiredis-1.0.2",
    url = "https://github.com/redis/hiredis/archive/refs/tags/v1.0.2.tar.gz",
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
