workspace(name = "projectmetis")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

# Imports PyBind11 Bazel plugin.
http_archive(
  name = "pybind11_bazel",
  sha256 = "43a2d54d833bba1c19ed04bfb0e09eb73a50513b300f654573637351656cd3ab",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/master.zip"],
  strip_prefix = "pybind11_bazel-master",
)

# Imports PyBind11 library.
http_archive(
  name = "pybind11",
  sha256 = "7a92b5c4433b445dadcafad99c95b399bd823af0430c73d0ca5da03a570a69dd",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-stable",
  urls = ["https://github.com/pybind/pybind11/archive/stable.tar.gz"],
)

# Configures MetisProject Python environment.
new_local_repository(
    name = "metis_python",
    path = "python/metisenv",
    build_file_content = """
exports_files(["bin/python", "bin/python3"])
filegroup(
    name = "files",
    srcs = glob(["**/*"], exclude = ["* *", "**/* *"]),
    visibility = ["//visibility:public"]
)"""
)

# Registers Python toolchain setup.
register_toolchains("//python:my_py_toolchain")

# Imports Python rules.
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.3.0/rules_python-0.3.0.tar.gz",
    sha256 = "934c9ceb552e84577b0faf1e5a2f0450314985b4d8712b2b70717dc679fdc01b",
)

# Defines Python package dependencies.
load("@rules_python//python:pip.bzl", "pip_install")
# Creates a central repo that knows about the dependencies needed for requirements.txt.
pip_install(
   name = "python_deps",
   requirements = "//:requirements.txt",
   python_interpreter_target = "@metis_python//:bin/python",
)
