workspace(name = "projectmetis")

load("//:GlobalVars.bzl", "METIS_VENV_PATH")

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
  sha256 = "3dc6435bd41c058453efe102995ef084d0a86b0176fd6a67a6b7100a2e9a940e",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/992381ced716ae12122360b0fbadbc3dda436dbf.zip"],
  strip_prefix = "pybind11_bazel-992381ced716ae12122360b0fbadbc3dda436dbf",
)

# Imports PyBind11 library.
http_archive(
  name = "pybind11",
  sha256 = "bcb738109172ec99ca7243bebe4617acbd7215dc5448741459911884263eba3d",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-stable",
  urls = ["https://github.com/pybind/pybind11/archive/stable.tar.gz"],
)

# Configures MetisProject Python environment.
new_local_repository(
    name = "metis_python",
    path = METIS_VENV_PATH,
    build_file_content = """
exports_files(["bin/python", "bin/python3"])
filegroup(
    name = "files",
    srcs = glob(["**/*"],
    exclude = ["* *", "**/* *"]),
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

load("@rules_python//python:pip.bzl", "pip_install")
# Creates a central repo that knows about the python interpreter dependencies based on requirements.txt.
pip_install(
   name = "python_deps",
   requirements = "//:requirements.txt",
   python_interpreter_target = "@metis_python//:bin/python",
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
# Imports Palisades library.
new_git_repository(
    name = "palisade_git",
    commit = "fded3711c5855e968467e2e73ccf2fcd7948dc7d",
    shallow_since = "1622060746 -0400",
    remote = "https://gitlab.com/palisade/palisade-development.git",
    verbose = True,
    recursive_init_submodules = True,
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)

# Imports SHELFI-FHE library.
new_git_repository(
    name = "shelfi_fhe_git",
    branch = "main",
    remote = "https://github.com/tanmayghai18/he-encryption-shelfi.git",
    verbose = True,
    build_file_content = """
filegroup(
    name = "all",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)
""",
)

# Imports foreign rules repository.
http_archive(
    name = "rules_foreign_cc",
    sha256 = "33a5690733c5cc2ede39cb62ebf89e751f2448e27f20c8b2fbbc7d136b166804",
    strip_prefix = "rules_foreign_cc-0.5.1",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.5.1.tar.gz"
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

# This sets up some common toolchains for building targets. For more details, please see
# https://bazelbuild.github.io/rules_foreign_cc/0.4.0/flatten.html#rules_foreign_cc_dependencies
rules_foreign_cc_dependencies()
